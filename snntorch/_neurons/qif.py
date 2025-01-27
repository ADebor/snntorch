from snntorch._neurons import SpikingNeuron

import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation


class MQIF(SpikingNeuron):
    """
    mQIF neuronal model adapted from "Drion G, Franci A, Seutin V, Sepulchre R. A novel phase portrait for neuronal excitability. PLoS One. 2012"
    ---
    model:
        dv/dt = v² - u² + I
        du/dt = ε(av - u + u_rest)
    ---
    discrete update equations:
        V[t+1] = V[t] (1 + V[t]) - U²[t] + WX[t]
        U[t+1] = βU[t] + (1 - β)(V[t] + U_rest)
    ---
    parameters:
        β = 1 - εa
        U_0
    """

    def __init__(
        self,
        eps,
        learn_eps,
        u_rest,
        learn_u_rest,
        v_init,
        u_init,
        v_reset=1.0,
        u_reset=1.0,
        threshold=1,
        a=0.1,
        dt=1,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        # inhibition=False,
        learn_threshold=False,
        # reset_mechanism="subtract",
        # state_quant=False,
        output=False,
        graded_spikes_factor=1,
        learn_graded_spikes_factor=False,
    ):
        super().__init__(
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            init_hidden=init_hidden,
            # inhibition,
            learn_threshold=learn_threshold,
            # reset_mechanism,
            # state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )
        self._init_state(v_init, u_init)
        self._mqif_register_buffer(
            eps,
            learn_eps,
            u_rest,
            learn_u_rest,
        )
        # self.v_reset = v_reset
        # self.u_reset = u_reset
        self.register_buffer(
            "dt", torch.tensor(dt)
        )
        self.register_buffer(
            "state_reset_values", torch.tensor([v_reset, u_reset])
        )
        self.register_buffer(
            "state_init_values", torch.tensor([v_init, u_init])
        )
        self.register_buffer("a", torch.tensor(a))

    def _mqif_register_buffer(self, beta, learn_beta, u_rest, learn_u_rest):
        self._eps_buffer(beta, learn_beta)
        self._u_rest_buffer(u_rest, learn_u_rest)

    def _eps_buffer(self, eps, learn_eps):
        if not isinstance(eps, torch.Tensor):
            eps = torch.as_tensor(eps)
        if learn_eps:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)

    def _u_rest_buffer(self, u_rest, learn_u_rest):
        if not isinstance(u_rest, torch.Tensor):
            u_rest = torch.as_tensor(u_rest)
        if learn_u_rest:
            self.u_rest = nn.Parameter(u_rest)
        else:
            self.register_buffer("u_rest", u_rest)

    # def state_function(self, input_):
    #     v = self.state[0]
    #     u = self.state[1]

    #     self.state[0] = (
    #         v * (1 + self.dt * v) - self.dt * u**2 + self.dt * input_
    #     )  # - self.reset * self.v_reset #TODO: reset not ok
    #     self.state[1] = (1 - self.dt * self.eps) * u + self.dt * self.eps * (
    #         self.a * v + self.u_rest
    #     )  # - self.reset * self.u_reset

    def state_function(input_, state, dt, eps, a, u_rest):
        v = state[0]
        u = state[1]

        state[0] = (
            v * (1 + dt * v) - dt * u**2 + dt * input_
        )  # - self.reset * self.v_reset #TODO: reset not ok
        state[1] = (1 - dt * eps) * u + dt * eps * (
            a * v + u_rest
        )  # - self.reset * self.u_reset

        return state

    @property
    def v_shift(self):
        return self.state[0] - self.threshold

    @property
    def reset(self):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        reset = self.spike_grad(self.v_shift).clone().detach()

        return reset

    @property
    def mem(self):
        return self.state[0]

    def fire(self):
        """Generates spike if v > threshold.
        Returns spk."""
        spk = self.spike_grad(self.v_shift)
        spk = spk * self.graded_spikes_factor

        return spk

    def forward(self, input_, state=None):
        print("input is of shape", input_.shape)
        "forward model with input current"
        if not state == None:
            self.state = state

        if self.init_hidden and not state == None:
            raise TypeError(
                "`state` should not be passed as an argument while `init_hidden=True`"
            )

        if not self.state.shape[1:] == input_.shape:
            self.state = (
                torch.zeros_like(input_, device=self.state.device)
                .unsqueeze(0)
                .expand(2, *input_.shape)
                .clone()
            )

        self.state_function(input_)
        spk = self.fire()  # TODO: useless in non-LIF model ?

        prev_state = self.state.clone().detach()
        reset = self.reset
        self.state -= self.reset * (self.state - self.state_reset_values)
        # if reset:
        #     print("prev", prev_state, "now", self.state)
        if self.output:
            return (
                spk,
                self.state,
                prev_state,
            )  # TODO: output state before or after reset ??
        elif self.init_hidden:
            return spk
        else:
            return spk, self.state, prev_state

    def _init_state(self, v_init, u_init):
        "init state variables"
        # state = torch.zeros(2, 0)
        state = torch.tensor([v_init, u_init])
        self.register_buffer("state", state, False)

    def reset_state(self):    #TODO: where should we reset the state?
        self.state = self.state_init_values.clone().detach()
        return self.state

    @classmethod
    def detach_hidden(cls): 
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], MQIF):
                cls.instances[layer].state.detach_()   

    @classmethod
    def reset_hidden(cls): #TODO: do we want hidden state to be zeroed on reset? 
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], MQIF):
                cls.instances[layer].state = cls.instances[layer].reset_state()
                # torch.zeros_like(
                #     cls.instances[layer].state,
                #     device=cls.instances[layer].state.device,
                # )

def plot_cur_mem_spk(
    cur,
    state,
    spk,
    u_rest, a,
    # cur_in, 
    v_init, u_init,
    thr_line=False,
    vline=0.,
    hline=0.,
    title=False,
    ylim_max1=1.25,
    ylim_max2=1.25,
    plot_traj=True,
):
    from pathlib import Path
    save_dir = Path(__file__).parent / 'figures'
    save_prefix = f"{save_dir}/{title}_"

    # Generate Plots
    fig, ax = plt.subplots(
        4,
        figsize=(8, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.4]},
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
    ax[1].plot(state[:, 0])
    # ax[1].set_ylim([0, ylim_max2])
    ax[1].set_ylabel("Membrane Potential ($V$)")
    if thr_line:
        ax[1].axhline(
            y=thr_line, alpha=0.25, linestyle="dashed", c="red", linewidth=1
        )

    # Plot membrane potential
    ax[2].plot(state[:, 1])
    # ax[1].set_ylim([0, ylim_max2])
    ax[2].set_ylabel("Membrane Potential ($U$)")

    plt.xlabel(f"Time (ms)")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[3], s=400, c="black", marker="|")
   
    plt.ylabel("Output spikes")
    plt.yticks([])
    # ax[3].xaxis.set_major_locator(ticker.MultipleLocator(25))

    ticks = np.arange(start=0, stop=cur.shape[0], step=1 / dt)
    plt.xticks(ticks=ticks, color="w")
    plt.savefig(save_prefix + "traces.png")
    plt.close()

    if not plot_traj:
        return

    # trajectory
    fig = plt.figure(figsize=(5,5), dpi=500)

    # x_min, x_max = state[:, 0].min() - 0.5,  state[:, 0].max() + 0.5
    # y_min, y_max = state[:, 1].min() - 0.05, state[:, 1].max() + 0.05
    # x_min, x_max, y_min, y_max = x_min.item(), x_max.item(), y_min.item(), y_max.item()

    x_max = torch.max(torch.abs(state[:, 0].min()), torch.abs(state[:, 0].max())).item() + 0.5
    x_min = -x_max
    if thr_line:
        x_max = np.max([thr_line, x_max]) + .5
    # y_max = torch.max(torch.abs(state[:, 1].min()), torch.abs(state[:, 1].max())).item() + 0.2
    # y_min = -y_max
    y_max = torch.sqrt(torch.abs(torch.max(cur))).item() + .5
    y_min = -y_max

    ax = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max))
    ax.set_xlabel("$V$")
    ax.set_ylabel("$U$")
    
    x = np.linspace(x_min, x_max, 100)
    plt.plot(x, a*x+u_rest, color="cyan", linewidth=1.5, alpha=0.2)
    
    (line,) = ax.plot([], [], lw=1, color='black', alpha=0.35)

    (line2,) = ax.plot([], [], lw=1.5, color='crimson', alpha=0.2)
    (line3,) = ax.plot([], [], lw=1.5, color='crimson', alpha=0.2)
    scat = ax.scatter([], [], s=2, alpha=1)

    ax.axvline(
        x=vline,
        ymin=y_min,
        ymax=y_max,
        alpha=0.5,
        linestyle="dashed",
        c="lightgray",
        linewidth=1,
        zorder=0,
        clip_on=True,
    )
    ax.axhline(
        y=hline,
        xmin=x_min,
        xmax=x_max,
        alpha=0.5,
        linestyle="dashed",
        c="lightgray",
        linewidth=1,
        zorder=0,
        clip_on=True,
    )
    if thr_line:
        ax.axvline(
            x=thr_line, alpha=0.5, linestyle="dashed", c="red", linewidth=1
        )
    
    # trajectories
    state = np.insert(state, 0, np.array((v_init, u_init)), 0)
    cur = np.insert(cur, 0, cur[0])

    def animate(n):
        line.set_xdata(state[:n+1, 0])
        line.set_ydata(state[:n+1, 1])


        cur_ = cur[n].item()

        line2.set_xdata(x)
        line2.set_ydata(np.sqrt(x**2 + cur_))
        line3.set_xdata(x)
        line3.set_ydata(-np.sqrt(x**2 + cur_))

        scat.set_offsets([state[n, 0], state[n, 1]])
        
        return (line,)

    anim = FuncAnimation(
        fig, animate, frames=state[:, 0].shape[0], interval=100
    )
    anim.save(save_prefix + "trajectory.gif")

 
if __name__ == "__main__":
    # Small step current input
    extra_cur = 0.55
    eps=.85
    u_rest=-.85
    threshold=4
    a=0.1
    v_reset=-0.
    u_reset=1.3
    v_init=-1. 
    u_init=-0.6
    dts=[
        0.001,
        0.01,
        0.05,
        0.1, 
        0.5, 
        0.75,
        1.,
    ]
    # input_type = 'step'
    input_type = 'impulse'

    for dt in dts:
        mqif = MQIF(
            eps=eps,
            learn_eps=False,
            u_rest=u_rest,
            learn_u_rest=False,
            v_reset=v_reset,
            u_reset=u_reset,
            v_init=v_init,
            u_init=u_init,
            threshold=threshold,
            a=a,
            dt=dt,
        )

        if input_type == 'impulse':
            cur_in = torch.cat(
                (
                    torch.zeros(int(np.ceil(10 / dt))) + extra_cur,
                    torch.ones(int(np.ceil(5 / dt))) * 1.,
                    torch.zeros(int(np.ceil(20 / dt))) + extra_cur,
                    torch.ones(int(np.ceil(5 / dt))) * -.5,
                    torch.zeros(int(np.ceil(10 / dt))) + extra_cur,
                ),
                0,
            )
        elif input_type == 'step':
            cur_in = torch.cat(
                (
                    torch.zeros(int(np.ceil(10 / dt))) + extra_cur,
                    torch.ones(int(np.ceil(30 / dt))) * 1.,
                    torch.zeros(int(np.ceil(10 / dt))) + extra_cur,
                ),
                0,
            )
        state = mqif.state
        num_steps = cur_in.shape[0]
        
        mem_rec = []
        spk_rec = []
        
        for step in range(num_steps):
            spk, state, prev_state = mqif(cur_in[step], state=state)
            mem_rec.append(
                # prev_state
                state.clone().detach()
            )
            spk_rec.append(spk)

        mem_rec = torch.stack(mem_rec)
        spk_rec = torch.stack(spk_rec)

        plot_cur_mem_spk(
            cur_in,
            mem_rec,
            spk_rec,
            thr_line=threshold,
            hline=0.,vline=0.,
            # ylim_max1=100,
            title=f"mqif_u_v_trajectory_{input_type}_{extra_cur}_{eps}_{u_rest}_{v_reset}_{u_reset}_{v_init}_{u_init}_{a}_{threshold}_{dt}",
            u_rest=u_rest, a=a, 
            # cur_in=cur_in, 
            v_init=v_init, u_init=u_init,
            plot_traj=False
        )

        mem_rec = []
        spk_rec = []
