import torch.nn as nn
from snntorch import spikegen


class RateCoding(nn.Module):
    def __init__(
        self, num_steps: float, gain: float = 0.5, num_silence_steps: int = 0
    ):
        super(RateCoding, self).__init__()
        self.num_steps = num_steps
        self.gain = gain
        self.num_silence_steps = num_silence_steps

    def forward(self, img):
        spks = spikegen.rate(img, num_steps=self.num_steps, gain=self.gain)
        if self.num_silence_steps == 0:
            return spks
        lim = spks.shape[0] - self.num_silence_steps
        if lim < 0:
            lim = spks.shape[0]
        spks[lim:] = 0.0
        return spks.transpose(
            0, 1
        )  # batch_size first to match torch.transforms output shape
