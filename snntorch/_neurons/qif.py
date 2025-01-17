from .neurons import SpikingNeuron
import torch
from torch import nn


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
        beta,
        learn_beta,
        u_rest,
        learn_u_rest,
        v_reset=1.,
        u_reset=1.,
        threshold=1,
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
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            # inhibition,
            learn_threshold,
            # reset_mechanism,
            # state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )
        self._init_state()
        self._mqif_register_buffer(
            beta,
            learn_beta,
            u_rest,
            learn_u_rest,
        )
        self.v_reset = v_reset
        self.u_reset = u_reset

    def _mqif_register_buffer(self, beta, learn_beta, u_rest, learn_u_rest):
        self._beta_buffer(beta, learn_beta)
        self._u_rest_buffer(u_rest, learn_u_rest)

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

    def _u_rest_buffer(self, u_rest, learn_u_rest):
        if not isinstance(u_rest, torch.Tensor):
            u_rest = torch.as_tensor(u_rest)
        if learn_u_rest:
            self.u_rest = nn.Parameter(u_rest)
        else:
            self.register_buffer("u_rest", u_rest)

    def state_function(self, input_):
        v = self.state[0]
        u = self.state[1]
        self.state[0] = v * (1 + v) - u**2 + input_ #- self.reset * self.v_reset #TODO: reset not ok
        self.state[1] = self.beta * u + (1 - self.beta) * (v + self.u_rest) #- self.reset * self.u_reset

    @property
    def v_shift(self):
        return self.state[0] - self.threshold
    
    @property
    def reset(self):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        reset = self.spike_grad(self.v_shift).clone().detach()

        return reset

    def fire(self):
        """Generates spike if v > threshold.
        Returns spk."""
        spk = self.spike_grad(self.v_shift)
        spk = spk * self.graded_spikes_factor

        return spk

    def forward(self, input_, state=None):
        "forward model with input current"
        if not state == None:
            self.state = state

        if self.init_hidden and not state == None:
            raise TypeError(
                "`state` should not be passed as an argument while `init_hidden=True`"
            )

        if not self.state.shape[1:] == input_.shape:
            self.state = torch.zeros_like(input_, device=self.mem.device)

        self.state_function(input_)
        spk = self.fire() #TODO: useless in non-LIF model ? 

        self.state -= self.reset * (self.state - self.state_reset_values) 

        if self.output:
            return spk, self.state  #TODO: output state before or after reset ??
        elif self.init_hidden:
            return spk
        else:
            return spk, self.state

    def _init_state(self):
        "init state variables"
        state = torch.zeros(0)
        self.register_buffer("state", state, False)
        
    def reset_state(self):  #TODO change name because also reset within step
        "init state variables"
        self.state = torch.zeros_like(self.state, device=self.state.device)
    



        