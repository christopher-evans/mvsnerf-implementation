import torch
import torch.nn as nn
import kornia.nerf.positional_encoder


class PositionalEncoding(nn.Module):
    def __init__(self, frequency_count=10, concat_inputs=False):
        super(PositionalEncoding, self).__init__()
        if frequency_count < 2:
            raise ValueError(f'Frequency count must be at least 2, received [{frequency_count}]')

        self.concat_inputs = concat_inputs

        # we use log sampling, so max frequency = frequency count - 1
        max_frequency = frequency_count - 1
        frequency_bands = 2.0 ** torch.linspace(0.0, max_frequency, steps=frequency_count)
        self.register_buffer('frequency_bands', frequency_bands.reshape(1, -1, 1))

    def forward(self, x):
        repeat = x.dim() - 1
        x_scaled = (
            x.unsqueeze(-2) * self.frequency_bands.view(*[1] * repeat, -1, 1)
        ).reshape(*x.shape[:-1], -1)

        if self.concat_inputs:
            return torch.cat(
                (x, torch.sin(x_scaled), torch.cos(x_scaled)), dim=-1
            )

        return torch.cat(
            (torch.sin(x_scaled), torch.cos(x_scaled)), dim=-1
        )
