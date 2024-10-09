from diffusers.utils import is_torch_available


if is_torch_available():
    from .unet_1d_condition import UNet1DConditionModel
