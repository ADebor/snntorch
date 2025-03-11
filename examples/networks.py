import torch
import torch.nn as nn
import snntorch as snn

from synapses import LeakyIntegratorSynapse, ScaledTanh
from utils import output_last

class LIFNet(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_hidden,
        n_outputs,
        neuron_params,
        n_steps,
        use_synapse=True,
    ):
        super().__init__()

        # Create layers
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.lif1 = snn.Leaky(**neuron_params, reset_mechanism="zero")
        self.fc2 = nn.Linear(n_hidden, n_outputs)
        self.lif2 = snn.Leaky(**neuron_params, reset_mechanism="zero")

        # Initialize layers
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        # # # Initialize biases
        # with torch.no_grad():
        #     self.fc1.bias = nn.Parameter(0. * torch.ones_like(self.fc1.bias), requires_grad=True)
        #     self.fc2.bias = nn.Parameter(0. * torch.ones_like(self.fc2.bias), requires_grad=True)

        self.n_steps = n_steps
        self.syn = use_synapse

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


class MQIFNet(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        layers: list,
        n_outputs: int,
        neuron_params: dict,
        use_synapse: bool=True,
    ):
        super().__init__()

        # Create layers
        layers_ = []
        layers.insert(0, n_inputs)
        layers.append(n_outputs)
        for _, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:])):
            
            # linear layer
            layers_.append(nn.Linear(n_in, n_out))
            nn.init.xavier_uniform_(layers_[-1].weight)
            nn.init.constant_(layers_[-1].bias, 0.0)

            # synapse
            if use_synapse: 
                layers_.append(
                    LeakyIntegratorSynapse(
                        feature_size=n_out, 
                        k_1=0.9,
                        k_2=0.8,
                        learn_params=False,
                    )
                )
            layers_.append(ScaledTanh(k_1=6., k_2=0.8, learn_params=True))

            # mqif
            layers_.append(output_last(snn.FastMQIF(**neuron_params)))

        # assembly
        self.model = nn.Sequential(*layers_)

    def forward(self, x):
        """Performs a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size x num_steps x n_inputs].
        Returns:
            torch.Tensor: Output tensor after passing through the model with shape [batch_size x num_steps x n_outputs].
        """

        output: torch.Tensor = self.model(x)
        # plt.plot(output[0].detach().cpu())
        # save_dir = Path(__file__).parent / "figures"
        # plt.savefig(f"{save_dir}/out.png")
        # plt.close()
    
        return output
