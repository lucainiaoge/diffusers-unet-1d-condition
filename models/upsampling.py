# Adapted from Huggingface by Tongyu Lu
# https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/upsampling.py

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.normalization import RMSNorm

# New
class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        if output_size is None:
            outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        else:
            outputs = F.interpolate(inputs, size=output_size, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs

# New
def upfirdn1d_native(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    up: int = 1,
    down: int = 1,
    pad: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    pad_0 = pad[0]
    pad_1 = pad[1]

    _, channel, in_len = tensor.shape
    tensor = tensor.reshape(-1, in_len, 1) # [B*C, L, 1]

    _, in_len, minor = tensor.shape
    kernel_len = len(kernel)

    out = tensor.view(-1, in_len, 1, minor) # [B*C, L, 1, 1]
    out = F.pad(out, [0, 0, 0, up - 1]) # [B*C, L, u, 1]
    out = out.view(-1, in_len * up, minor) # [B*C, uL, 1]

    out = F.pad(out, [0, 0, max(pad_0, 0), max(pad_1, 0)])
    out = out.to(tensor.device)  # Move back to mps if necessary
    out = out[:, max(-pad_0, 0) : out.shape[1] - max(-pad_1, 0), :] # [B*C, uL(padded), 1]

    out = out.permute(0, 2, 1) # [B*C, 1, uL(padded)]
    out = out.reshape([-1, 1, in_len * up + pad_0 + pad_1])
    w = torch.flip(kernel, [0,]).view(1, 1, kernel_len) # [1, 1, K]
    out = F.conv1d(out, w)
    out = out.reshape(-1, minor, in_len * up + pad_0 + pad_1 - kernel_len + 1)
    out = out.permute(0, 2, 1) # [B*C, uL(conved), 1]
    out = out[:, ::down, :] # [B*C, out_len, 1]

    out_len = (in_len * up + pad_0 + pad_1 - kernel_len) // down + 1

    return out.view(-1, channel, out_len)

# New
def upsample_1d(
    hidden_states: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> torch.Tensor:
    r"""Upsample1D a batch of 1D seqs with the given filter.
    Accepts a batch of 1D seqs of the shape `[N, C, L]` (thanks to reshape method) and upsamples each seq with the
    given filter. The filter is normalized so that if the inputs are constant, they will be scaled by the
    specified `gain`. Samples outside the seq are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the upsampling factor.

    Args:
        hidden_states (`torch.Tensor`):
            Input tensor of the shape `[N, C, L]` or `[N, L, C]`.
        kernel (`torch.Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to nearest-neighbor upsampling.
        factor (`int`, *optional*, default to `2`):
            Integer upsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output (`torch.Tensor`):
            Tensor of the shape `[N, C, L * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    assert kernel.ndim == 1
    kernel /= torch.sum(kernel)

    kernel = kernel * (gain * factor)
    pad_value = kernel.shape[0] - factor
    output = upfirdn1d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output