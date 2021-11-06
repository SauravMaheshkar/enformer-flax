from typing import Sequence

from flax import linen as nn
from flax import struct

from enformer_flax.layers.container_layers import Residual, Sequential
from enformer_flax.layers.convolution_layers import ConvBlock
from enformer_flax.layers.pooling_layers import SoftmaxPooling1D

__all__ = ["Stem", "ConvTower", "TransformerMLP", "Transformer", "FinalPointwise"]


# ================ Stem ===================


@struct.dataclass
class Stem(nn.Module):

    channels: int
    pool_size: int = 2
    kernel_size: int = 15

    def setup(self):
        self.layers = [
            nn.Conv(features=self.channels // 2, kernel_size=self.kernel_size),
            Residual([ConvBlock(filters=self.channels // 2)]),
            SoftmaxPooling1D(pool_size=self.pool_size),
        ]

    @nn.compact
    def __call__(self, inputs):
        return Sequential(self.layers)(inputs)


# ================ ConvTower ===================


@struct.dataclass
class ConvTower(nn.Module):

    filter_list: Sequence
    pool_size: int = 2

    def setup(self):
        self.block = Sequential(
            [
                ConvBlock(filters=num_filters, width=5),
                Residual([ConvBlock(filters=num_filters, width=1)]),
                SoftmaxPooling1D(pool_size=self.pool_size),
            ]
            for i, num_filters in enumerate(self.filter_list)
        )

    @nn.compact
    def __call__(self, inputs):
        return Sequential(self.block)(inputs)


# ================ TransformerMLP ===================


@struct.dataclass
class TransformerMLP(nn.Module):

    channels: int
    dropout_rate: float = 0.4

    def setup(self):
        self.layers = [
            nn.LayerNorm(use_scale=True, use_bias=True),
            nn.Dense(features=self.channels * 2),
            nn.Dropout(rate=self.dropout_rate),
            nn.relu(),
            nn.Dense(features=self.channels),
            nn.Dropout(rate=self.dropout_rate),
        ]

    @nn.compact
    def __call__(self, inputs):
        return Sequential(self.layers)(inputs)


# ================ Transformer ===================


@struct.dataclass
class Transformer(nn.Module):

    channels: int
    num_transformer_layers: int
    num_heads: int = 8
    dropout_rate: float = 0.4
    attention_dropout_rate: float = 0.05

    def setup(self):
        self.layers = Sequential(
            [
                Residual(
                    [
                        Sequential(
                            [
                                nn.LayerNorm(use_bias=True, use_scale=True),
                                nn.MultiHeadDotProductAttention(
                                    num_heads=self.num_heads,
                                    dropout_rate=self.attention_dropout_rate,
                                ),
                                nn.Dropout(rate=self.dropout_rate),
                            ]
                        )
                    ]
                ),
                Residual([TransformerMLP(self.channels)]),
            ]
            for _ in range(self.num_transformer_layers)
        )

    @nn.compact
    def __call__(self, inputs):
        return Sequential(self.layers)(inputs)


# ================ FinalPointwise ===================


@struct.dataclass
class FinalPointwise(nn.Module):

    channels: int
    dropout_rate: float = 0.4

    def setup(self):
        self.layers = [
            ConvBlock(filters=self.channels * 2, width=1),
            nn.Dropout(rate=self.dropout_rate),
            nn.gelu(),
        ]

    @nn.compact
    def __call__(self, inputs):
        return Sequential(self.layers)(inputs)
