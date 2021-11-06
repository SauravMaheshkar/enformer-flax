from typing import Sequence, Type

from flax import linen as nn
from flax import struct

__all__ = ["Residual", "Sequential"]

# ================ Residual ===================


@struct.dataclass
class Residual(nn.Module):
    layers: Sequence[Type[nn.Module]]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x


# ================ Sequential ===================


@struct.dataclass
class Sequential(nn.Module):
    layers: Sequence[Type[nn.Module]]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
