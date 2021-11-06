from typing import Callable

import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from jax import dtypes
from jax.experimental.djax import reduce_sum

__all__ = ["identity", "SoftmaxPooling1D"]

# ================ Identity Initializer ===================


def identity(shape, gain: float = 1.0, dtype=jnp.float_):
    return gain * jnp.eye(*shape, dtype=dtypes.canonicalize_dtype(dtype))


# ================ SoftmaxPooling1D ===================


@struct.dataclass
class SoftmaxPooling1D(nn.Module):

    pool_size: int = 2
    per_channel: bool = False
    w_init_scale: float = 0.0

    @nn.nowrap
    def _initialize(self, num_features: int) -> Callable:
        return nn.Dense(
            features=num_features,
            use_bias=False,
            kernel_init=identity(
                shape=(num_features, num_features), gain=self.w_init_scale
            ),
        )

    @nn.compact
    def __call__(self, inputs):
        _, length, num_features = inputs.shape
        self._logit_linear = self._initialize(num_features)

        inputs = jnp.reshape(
            a=inputs,
            newshape=(-1, length // self._pool_size, self._pool_size, num_features),
        )

        return reduce_sum(
            inputs * nn.softmax(self._logit_linear(inputs), axis=-2), axes=-2
        )
