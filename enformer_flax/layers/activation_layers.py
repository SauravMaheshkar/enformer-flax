import jax
from flax import linen as nn
from flax import struct

__all__ = ["SoftPlus", "GELU"]

# ================ SoftPlus ===================


@struct.dataclass
class SoftPlus(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jax.nn.softplus(x)


# ================ GELU ===================


@struct.dataclass
class GELU(nn.Module):
    approximate: bool = False

    @nn.compact
    def __call__(self, x):
        return jax.nn.gelu(x, approximate=self.approximate)
