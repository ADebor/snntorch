import torch
import torch.nn as nn
import snntorch as snn


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

        self.mqif1 = snn.MQIF(**neuron_params)

        self.fc2 = nn.Linear(n_hidden, n_outputs)

        self.mqif2 = snn.MQIF(**neuron_params)

        # Initialize layers
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        # stdv = 1. / math.sqrt(self.fc1.weight.size(1))
        # self.fc1.weight.data.uniform_(-.5*stdv, 1.*stdv)

        # stdv = 1. / math.sqrt(self.fc2.weight.size(1))
        # self.fc2.weight.data.uniform_(-.75*stdv, 1.5*stdv)

        # Initialize biases
        with torch.no_grad():
            self.fc1.bias = nn.Parameter(
                0.0 * torch.ones_like(self.fc1.bias), requires_grad=False
            )
            self.fc2.bias = nn.Parameter(
                0.0 * torch.ones_like(self.fc2.bias), requires_grad=False
            )

        self.n_steps = n_steps
        self.syn = use_synapse

        # create synapses
        self.register_buffer("syn1", None)
        self.register_buffer("syn2", None)
        # self.hardtanh = torch.nn.Hardtanh(2., self.mqif1.state_reset_values[:, 1].item()**2 - 0.3)

    def reset_synapses(self, batch_size):
        self.syn1 = torch.zeros((batch_size, self.fc1.out_features)).cuda()
        self.syn2 = torch.zeros((batch_size, self.fc2.out_features)).cuda()

    def forward(self, x):
        """x.shape: [batch_size x num_steps x input dimensions]"""

        # Initialize hidden states at t=0
        state1 = self.mqif1.reset_state()
        state2 = self.mqif2.reset_state()

        # Record the final layer
        spk2_rec = []
        state2_rec = []
        state1_rec = []
        cur1_rec = []
        cur2_rec = []
        syn1_rec = []
        syn2_rec = []

        # reset synapses
        self.reset_synapses(batch_size=x.shape[0])

        for step in range(self.n_steps):
            # FC 1
            cur1 = self.fc1(x[:, step, :])

            # synapse 1
            k_1 = self.mqif1.state_reset_values[:, 1] ** 2 - 0.3
            if self.syn:
                # self.syn1 = self.hardtanh(0.9 * self.syn1 + 1.0 * cur1)
                # self.syn1 = k_1 * torch.tanh(
                #     (0.9 * self.syn1 + 1.0 * cur1) / k_1
                # )

                self.syn1 = 0.8 * self.syn1 + 1.0 * cur1

                # self.syn1 = k_1 * torch.sigmoid(
                #     (0.9 * self.syn1 + 1.0 * cur1) / k_1
                # )
            else:
                self.syn1 = k_1 * torch.tanh(cur1 / k_1)

            # mqif 1
            spk1, state1, _ = self.mqif1(
                (6.0 * torch.tanh(self.syn1 / 6.0)) ** 2, state1
            )  # TODO: not ok cause positive sign only !!!! (squared)
            # spk1, state1, _ = self.mqif1((1.*torch.tanh(self.syn1 / 10.) + 5.)**2, state1)
            # spk1, state1, _ = self.mqif1((1.*torch.tanh(self.syn1 / 10.) + 4.3)**2, state1)

            # FC 2
            cur2 = self.fc2(spk1)

            # synapse 2
            k_2 = self.mqif2.state_reset_values[:, 1] ** 2 - 0.3
            if self.syn:
                # HardTanh
                # self.syn2 = self.hardtanh(0.9 * self.syn2 + 1.0 * cur2)

                # Scaled Tanh
                # self.syn2 = k_2 * torch.tanh(
                #     (0.9 * self.syn2 + 1.0 * cur2) / k_2
                # )

                # Bistable Tanh
                # self.syn2 = torch.tanh(0.9 * self.syn2 + 1. * cur2)
                self.syn2 = 0.8 * self.syn2 + 0.9 * cur2

                # Sigmoid
                # self.syn2 = k_2 * torch.sigmoid(
                #     (0.9 * self.syn2 + 1.0 * cur2) / k_2
                # )
            else:
                self.syn2 = k_2 * torch.tanh(cur2 / k_2)

            # mqif 2
            # spk2, state2, _ = self.mqif2((1.*torch.tanh((self.syn2 + 0.3) / 5.) + 5.)**2, state2)
            # spk2, state2, _ = self.mqif2((torch.tanh((self.syn2)) + 4.3)**2, state2)
            spk2, state2, _ = self.mqif2(
                (6.0 * torch.tanh(self.syn2 / 6.0)) ** 2, state2
            )

            spk2_rec.append(spk2)

            state2_rec.append(state2.clone())
            state1_rec.append(state1.clone().detach())

            cur1_rec.append(cur1.clone().detach())
            cur2_rec.append(cur2.clone().detach())

            syn1_rec.append(
                (6.0 * torch.tanh(self.syn1.clone().detach() / 6.0)) ** 2
            )
            syn2_rec.append(
                (6.0 * torch.tanh(self.syn2.clone().detach() / 6.0)) ** 2
            )
            # syn1_rec.append((torch.tanh(self.syn1.clone().detach() / 10.) + 4.3)**2)
            # syn2_rec.append((torch.tanh(self.syn2.clone().detach()) + 4.3)**2)

        # # Plots
        # mem1_stack = torch.stack(state1_rec, dim=0)
        # plt.plot(mem1_stack[:, 0, :10, 0].detach().cpu(), label=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        # plt.title(f"mem1 v traces - 5 first neurons")
        # plt.ylim([-5., 10.])
        # plt.legend()
        # plt.savefig("./mem1_trace_v.png")
        # plt.close()

        # mem2_stack = torch.stack(state2_rec, dim=0)
        # plt.plot(mem2_stack[:, 0, :, 0].detach().cpu())
        # plt.title(f"mem2 v traces - all neurons")
        # plt.ylim([-4.5, 8.])
        # plt.savefig("./mem2_trace_v.png")
        # plt.close()

        # syn1_stack = torch.stack(syn1_rec, dim=0)
        # plt.plot(syn1_stack[:, 0, :].detach().cpu())
        # plt.title(f"syn1 traces")
        # # plt.ylim([-3.5, 3.5])
        # plt.savefig("./syn1_trace.png")
        # plt.close()

        # syn2_stack = torch.stack(syn2_rec, dim=0)
        # plt.plot(syn2_stack[:, 0, :].detach().cpu())
        # plt.title(f"syn2 traces")
        # # plt.ylim([-3.5, 3.5])
        # plt.savefig("./syn2_trace.png")
        # plt.close()

        # cur1_stack = torch.stack(cur1_rec, dim=0)
        # plt.plot(cur1_stack[:, 0, :].detach().cpu())
        # plt.title(f"cur1 traces")
        # # plt.ylim([-3., 3.2])
        # plt.savefig("./cur1_trace.png")
        # plt.close()

        # cur2_stack = torch.stack(cur2_rec, dim=0)
        # plt.plot(cur2_stack[:, 0, :].detach().cpu())
        # plt.title(f"cur2 traces")
        # # plt.ylim([-0.75, 0.75])
        # plt.savefig("./cur2_trace.png")
        # plt.close()
        # fig, ax = plt.subplots(
        #     1,
        #     figsize=(10, 10),
        #     # sharex=False,
        #     # gridspec_kw={"height_ratios": [1, 1, 2]},
        #     dpi=300,
        # )
        # splt.raster(torch.stack(spk2_rec, dim=0)[:, 0, :], ax, s=100, c="black", marker="|")
        # plt.xlim([0, 149])
        # plt.savefig("./spks_out.png")
        # plt.close()
        return torch.stack(spk2_rec, dim=0), torch.stack(state2_rec, dim=0)
