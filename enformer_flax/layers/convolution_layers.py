from typing import Callable

import jax
from flax import linen as nn
from flax import struct

__all__ = ["ConvBlock"]

# ================ ConvBlock ===================


@struct.dataclass
class ConvBlock(nn.Module):

    filters: int
    width: int = 1
    act: Callable = nn.gelu
    w_init = None

    def setup(self):
        self.bn = nn.BatchNorm(
            momentum=0.9,
            use_bias=True,
            use_scale=True,
            scale_init=jax.nn.initializers.ones(),
        )
        self.conv = nn.Conv(
            features=self.filters, kernel_size=self.width, kernel_init=self.w_init
        )

    @nn.compact
    def __call__(self, inputs):
        norm = self.bn(inputs)
        act_out = self.act(norm)
        return self.conv(act_out)
