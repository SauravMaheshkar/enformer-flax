from flax import linen as nn
from flax import struct

__all__ = ["TargetLengthCrop1D"]

# ================ TargetLengthCrop1D ===================


@struct.dataclass
class TargetLengthCrop1D(nn.Module):

    target_length: int

    @nn.compact
    def __call__(self, inputs):
        trim = (inputs.shape[-2] - self._target_length) // 2
        if trim < 0:
            raise ValueError("inputs longer than target length")

        return inputs[..., trim:-trim, :]
