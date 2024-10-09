from dataclasses import dataclass

from diffusers.utils import BaseOutput

@dataclass
class Transformer1DModelOutput(BaseOutput):
    """
    The output of [`Transformer1DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, length)`):
            The hidden states output conditioned on the `encoder_hidden_states` input.
    """

    sample: "torch.Tensor"