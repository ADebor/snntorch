# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch.nn as nn
import torch.optim

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder

import numpy as np
from snntorch import spikegen
import snntorch as snn
from snntorch import spikeplot as splt
import matplotlib.pyplot as plt

from pathlib import Path

from torchrl.envs import ParallelEnv, EnvCreator

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(
    env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False
):
    env = GymEnv(
        env_name, device=device, from_pixels=from_pixels, pixels_only=False
    )
    # env = TransformedEnv(env)
    # env.append_transform(
    #     VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2)
    # )
    # env.append_transform(
    #     ClipTransform(in_keys=["observation"], low=-10, high=10)
    # )
    # env.append_transform(RewardSum())
    # env.append_transform(StepCounter())
    # env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


def make_parallel_env(env_name, num_envs, device, from_pixels: bool = False):
    env = ParallelEnv(
        num_envs,
        EnvCreator(
            lambda: make_env(env_name, device="cpu", from_pixels=from_pixels)
        ),
        serial_for_single=False,
        device=device,
    )

    env = TransformedEnv(env)

    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.9, eps=1e-2))
    env.append_transform(
        ClipTransform(in_keys=["observation"], low=-10, high=10)
    )
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


class LIFNet(nn.Module):
    def __init__(
        self, n_inputs, n_hidden, n_outputs, lif_params, n_steps, syn=True
    ):
        super().__init__()

        # Create layers
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.lif1 = snn.Leaky(**lif_params, reset_mechanism="zero")
        self.fc2 = nn.Linear(n_hidden, n_outputs)
        self.lif2 = snn.Leaky(**lif_params, reset_mechanism="zero")

        # Initialize layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.n_steps = n_steps
        self.syn = syn

        # create synapses
        self.register_buffer("syn1", None)
        self.register_buffer("syn2", None)

    def reset_synapses(self, batch_size):
        self.syn1 = torch.zeros((batch_size, self.fc1.out_features)).cuda()
        self.syn2 = torch.zeros((batch_size, self.fc2.out_features)).cuda()

    def forward(self, x):
        """x.shape: [batch_size x num_steps x input dimensions]"""

        # Initialize hidden states at t=0
        state1 = self.lif1.reset_mem()
        state2 = self.lif2.reset_mem()

        # Record the final layer
        spk2_rec = []
        state2_rec = []

        # reset synapses
        self.reset_synapses(batch_size=x.shape[0])

        for step in range(self.n_steps):
            cur1 = self.fc1(x[:, step, :])
            k_1 = 2.0**2 - 0.3
            if self.syn:
                self.syn1 = k_1 * torch.tanh(
                    (0.9 * self.syn1 + 1.0 * cur1) / k_1
                )
            else:
                self.syn1 = cur1

            spk1, state1 = self.lif1(self.syn1, state1)

            cur2 = self.fc2(spk1)

            k_2 = 2.0**2 - 0.3
            if self.syn:
                self.syn2 = k_2 * torch.tanh(
                    (0.9 * self.syn2 + 1.0 * cur2) / k_2
                )
            else:
                self.syn2 = cur2
            spk2, state2 = self.lif2(self.syn2, state2)

            spk2_rec.append(spk2)

            state2_rec.append(state2.clone())

        return torch.stack(spk2_rec, dim=0), torch.stack(state2_rec, dim=0)


class MinplusLinear(nn.Module):
    def __init__(self, n_inputs, n_hidden, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "pos_weight",
            torch.rand((n_inputs // 2, n_hidden), requires_grad=False),
        )
        self.pos_weight = self._buffers["pos_weight"]
        self.register_buffer(
            "neg_weight",
            -1.0
            * torch.rand(
                (int(np.ceil(n_inputs / 2)), n_hidden), requires_grad=False
            ),
        )
        self.neg_weight = self._buffers["neg_weight"]

    def forward(self, x):
        # assume x is [batch_size, time, feature_size] -> output is [batch_size, time, 1]
        lim = x.shape[-1] // 2
        return (
            x[:, :, :lim] @ self.pos_weight + x[:, :, lim:] @ self.neg_weight
        )


class LeakyIntegratorSynapse(nn.Module):
    def __init__(self, feature_size, k_1=0.9, k_2=0.9, learn_params=False):
        """
        Args:
            feature_size (int): Number of features (last dimension of input tensor).
        """
        super().__init__()
        # Learnable parameters, initialized to 0.9
        self.k1 = nn.Parameter(
            torch.full((1, 1, feature_size), k_1), requires_grad=learn_params
        )
        self.k2 = nn.Parameter(
            torch.full((1, 1, feature_size), k_2), requires_grad=learn_params
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, time_dim, feature_size].

        Returns:
            torch.Tensor: Integrated output of the same shape.
        """
        batch_size, time_dim, feature_size = x.shape

        # Initialize output with zeros (first output is always zero)
        out = torch.zeros_like(x)

        # Compute integration
        coeffs = torch.cumprod(
            self.k1.expand(batch_size, time_dim, feature_size), dim=1
        )
        weighted_inputs = self.k2 * x
        out[:, 1:, :] = torch.cumsum(
            coeffs[:, :-1, :] * weighted_inputs[:, 1:, :], dim=1
        )
        return out


class GaussianReceptiveFieldEncoder(nn.Module):
    def __init__(
        self,
        observation_size: int,
        n_fields: int,
        n_steps: int,
        sigma: float = 1.0,
        learn_params: bool = False,
    ):
        """
        Args:
            observation_size (int): The size of each observation.
            n_fields (int): Number of Gaussian receptive fields.
            n_steps (int): Number of time steps in the spike train.
            sigma (float): Standard deviation of the Gaussian receptive fields.
        """
        super().__init__()
        self.observation_size = observation_size
        self.n_fields = n_fields
        self.n_steps = n_steps

        # Define equally spaced means from -3 to 3
        self.means = nn.Parameter(
            torch.linspace(-3.0, 3.0, n_fields).repeat(
                observation_size, 1
            ),  # Shape: [1, 1, n_fields]
            requires_grad=learn_params,
        )
        self.sigma = nn.Parameter(
            torch.Tensor([sigma]), requires_grad=learn_params
        )

    @property
    def output_dim(self) -> int:
        return self.n_fields * self.observation_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, observation_size]

        Returns:
            Tensor: Spike train of shape [batch_size, n_steps, observation_size, n_fields]
        """
        # batch_size, obs_size = x.shape

        # reshape input for broadcasting: [batch_size, observation_size, 1]
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.n_fields)

        # compute Gaussian probability density function values
        pdf_values = torch.exp(
            -0.5 * ((x_expanded - self.means) / self.sigma) ** 2
        )
        pdf_values /= self.sigma * torch.sqrt(torch.tensor(2 * torch.pi))

        # generate Bernoulli spike trains over n_steps
        spike_trains = spikegen.rate(
            pdf_values, num_steps=self.n_steps
        ).transpose(0, 1)

        # reshape: [batch_size, n_steps, observation_size*n_fields]
        shape = list(spike_trains.shape)
        shape[-2] *= shape[-1]
        shape.pop(-1)
        spike_trains = spike_trains.reshape(shape)

        return spike_trains


class CoupledTanhEncoder(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_steps: int,
        gain: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        self.gain = gain
        self.observation_size = observation_size

    @property
    def output_dim(self):
        return 2 * self.observation_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        pos_input = spikegen.rate(x, num_steps=self.num_steps, gain=self.gain)
        neg_input = spikegen.rate(-x, num_steps=self.num_steps, gain=self.gain)

        # reshape: [batch_size, n_steps, observation_size*2]
        output = torch.cat((pos_input, neg_input), dim=-1).transpose(0, 1)

        # fig = plt.figure(facecolor="w", figsize=(10, 5))
        # ax = fig.add_subplot(111)
        # splt.raster(output[0, :, :], ax=ax, s=400, c="black", marker="|")
        # save_dir = Path(__file__).parent / "figures"
        # fname = f"{save_dir}/coupled_input_spks.png"
        # plt.savefig(fname)

        return output


# def print_input(func):
#     def wrapper(input_, input__):
#         print("input is", input_, "and ", input__)
#         return func(input)
#     return wrapper


def output_last(obj):
    class Wrapper(nn.Module):
        def __init__(self, obj, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.obj = obj

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.obj(x)
            return out[-1]

    return Wrapper(obj)


def plot_cur_mem_spk(
    cur,
    v,
    u,
    spk,
    title,
    thr_line=False,
):

    save_dir = Path(__file__).parent / "figures"
    save_prefix = f"{save_dir}/{title}_"
    cur, v, u, spk = (
        cur[0].detach().cpu(),
        v[0].detach().cpu(),
        u[0].detach().cpu(),
        spk[0].detach().cpu(),
    )

    # Generate Plots
    fig, ax = plt.subplots(
        4,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [0.25, 0.25, 0.25, 1]},
        dpi=500,
    )

    # Plot input current
    ax[0].plot(cur, c="tab:orange")
    # ax[0].set_ylim([0, ylim_max1])
    # ax[0].set_xlim([0, 200])
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    if title:
        ax[0].set_title(title)

    # Plot membrane potential
    ax[1].plot(v)
    # ax[1].set_ylim([0, ylim_max2])
    ax[1].set_ylabel("Membrane Potential ($V$)")
    if thr_line:
        ax[1].axhline(
            y=thr_line, alpha=0.25, linestyle="dashed", c="red", linewidth=1
        )

    # Plot membrane potential
    ax[2].plot(u)

    # ax[1].set_ylim([0, ylim_max2])
    ax[2].set_ylabel("Membrane Potential ($U$)")

    plt.xlabel(f"Time (ms)")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[3], s=25, c="black", marker="|")
    plt.ylim((0, spk.shape[-1] - 1))
    plt.ylabel("Output spikes")
    # plt.yticks([])

    ticks = np.arange(start=0, stop=cur.shape[0], step=100)
    plt.xticks(ticks=ticks, color="w")
    plt.savefig(save_prefix + "traces.png")
    plt.close()


def plot_mqif(mqif, index):
    class Wrapper(nn.Module):
        def __init__(self, mqif, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mqif = mqif

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v, u, s = self.mqif(x)
            plot_cur_mem_spk(x, v, u, s, title=f"MQIF_{index}_traces")
            return v, u, s

    return Wrapper(mqif)


class ScaledTanh(nn.Tanh):
    def __init__(
        self,
        k_1: float,
        k_2: float = None,
        learn_params: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not k_2:
            k_2 = k_1
        self.k_1 = nn.Parameter(
            torch.Tensor([k_1]), requires_grad=learn_params
        )
        self.k_2 = nn.Parameter(
            torch.Tensor([k_2]), requires_grad=learn_params
        )

    def forward(self, input):
        return self.k_1 * super().forward(input / self.k_2)


class MQIFNet(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_steps: int,
        layers: list,
        n_outputs: int,
        mqif_params: dict,
        syn: bool = True,
    ):
        super().__init__()

        # observation encoder
        layers_ = [
            # GaussianReceptiveFieldEncoder(
            #     observation_size=n_inputs,
            #     n_fields=10,
            #     n_steps=n_steps,
            #     sigma=0.3,
            #     learn_params=False,
            # )
            CoupledTanhEncoder(
                observation_size=n_inputs,
                num_steps=n_steps,
                gain=0.25,
            ),
        ]
        layers.insert(0, layers_[0].output_dim)

        syn_k2 = [0.5, 0.1]
        for i in range(len(layers) - 1):
            # linear
            layers_.append(nn.Linear(layers[i], layers[i + 1]))
            nn.init.xavier_uniform_(layers_[-1].weight)
            nn.init.constant_(layers_[-1].bias, 0.0)

            # synapse
            if syn:
                layers_.append(
                    LeakyIntegratorSynapse(
                        feature_size=layers[i + 1],
                        k_1=0.9,
                        k_2=0.8,
                        learn_params=False,
                    )
                )
            layers_.append(
                ScaledTanh(k_1=6.0, k_2=syn_k2[i], learn_params=True)
            )

            # MQIF
            layers_.append(
                output_last(
                    # plot_mqif(
                    snn.FastMQIF(**mqif_params),
                    # index=i,
                    # )
                )
            )

        # action decoder
        layers_.extend(
            [
                MinplusLinear(layers[-1], n_outputs),
                LeakyIntegratorSynapse(
                    feature_size=n_outputs,
                    k_1=0.9,
                    k_2=1.0,
                    learn_params=False,
                ),
                ScaledTanh(k_1=3.0, k_2=0.5, learn_params=True),
            ]
        )

        # assembly
        self.model = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor):
        # TODO: check input normalization
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # x = spikegen.rate(
        #         torch.sigmoid(x / .5), # scaled tanh to have positive, scaled inputs
        #         num_steps=self.n_steps
        #         )
        # x = torch.transpose(x, 0, 1)    # batch first

        # compute integrator trace
        output: torch.Tensor = self.model(x)
        # plt.plot(output[0].detach().cpu())
        # save_dir = Path(__file__).parent / "figures"
        # plt.savefig(f"{save_dir}/out.png")
        # plt.close()
        # get last voltage value
        output = output[:, -1, :].squeeze(0)

        return output


def make_ppo_models_state(proof_environment, device, net_cfg, neuron_cfg):
    neuron_type = neuron_cfg.name
    if neuron_type not in ["ann", "mqif", "lif"]:
        exit("Invalid net type.")

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec_unbatched.shape[-1]

    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low.to(device),
        "high": proof_environment.action_spec_unbatched.space.high.to(device),
        "tanh_loc": False,
    }

    # Define policy architecture
    if neuron_type == "ann":
        policy_mlp = MLP(
            in_features=input_shape[-1],
            activation_class=nn.Tanh,
            out_features=num_outputs,  # predict only loc
            num_cells=[64, 64],
            device=device,
        )

        # Initialize policy weights
        for layer in policy_mlp.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 1.0)
                layer.bias.data.zero_()

    elif neuron_type == "mqif":
        neuron_params = neuron_cfg.params
        policy_mlp = MQIFNet(
            n_inputs=input_shape[-1],
            layers=net_cfg.policy.layers,
            n_outputs=num_outputs,
            mqif_params=neuron_params,
            n_steps=net_cfg.policy.n_steps,
            syn=net_cfg.policy.use_synapse,
        ).to(device)
    elif neuron_type == "lif":
        neuron_params = dict(beta=0.9, threshold=1.0)
        policy_mlp = LIFNet(
            n_inputs=input_shape[-1],
            n_hidden=256,
            n_outputs=10,
            lif_params=neuron_params,
            n_steps=net_cfg.policy.n_steps,
            syn=net_cfg.policy.use_synapse,
        ).to(device)

    # Add state-independent normal scale
    policy_mlp = nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            proof_environment.action_spec_unbatched.shape[-1], scale_lb=1e-8
        ).to(device),
    )

    # Add probabilistic sampling of the actions
    # print(proof_environment.action_spec_unbatched.to(device))
    # exit(proof_environment.full_action_spec_unbatched.to(device))
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
        device=device,
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module


def make_ppo_models(env_name, device, net_cfg, neuron_cfg):
    proof_environment = make_parallel_env(env_name, 1, device=device)
    # proof_environment = make_env(env_name, device=device)

    actor, critic = make_ppo_models_state(
        proof_environment,
        device=device,
        net_cfg=net_cfg,
        neuron_cfg=neuron_cfg,
    )

    # with torch.no_grad():
    #     td = proof_environment.fake_tensordict().expand(10)
    #     td = actor(td)
    #     critic(td)
    #     print(td)
    #     del td

    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
        test_env.apply(dump_video)
    del td_test
    return torch.cat(test_rewards, 0).mean()
