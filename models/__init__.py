from typing import TYPE_CHECKING

from diffusers.utils import (
    DIFFUSERS_SLOW_IMPORT,
    _LazyModule,
    is_flax_available,
    is_torch_available,
)


_import_structure = {}

if is_torch_available():
    _import_structure["unets.unet_1d_condition"] = ["UNet1DConditionModel"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .unets import UNet1DConditionModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)