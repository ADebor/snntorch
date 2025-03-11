import torch
import torch.nn as nn


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