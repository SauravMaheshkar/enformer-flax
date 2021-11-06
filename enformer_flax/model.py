from flax import linen as nn

from enformer_flax.layers.activation_layers import SoftPlus
from enformer_flax.layers.container_layers import Sequential
from enformer_flax.layers.misc import TargetLengthCrop1D
from enformer_flax.modules import ConvTower, FinalPointwise, Stem, Transformer
from enformer_flax.ops import exponential_linspace_int

TARGET_LENGTH = 896

__all__ = ["Enformer"]

# ================ Enformer ===================


class Enformer(nn.Module):
    channels: int = 1536
    num_transformer_layers: int = 11
    num_heads: int = 8

    def setup(self):
        heads_channels = {"human": 5313, "mouse": 1643}
        assert self.channels % self.num_heads == 0, (
            "channels needs to be divisible " f"by {self.num_heads}"
        )

        filter_list = exponential_linspace_int(
            start=self.channels // 2, end=self.channels, num=6, divisible_by=128
        )

        self.crop_final = TargetLengthCrop1D(TARGET_LENGTH)

        self._trunk = [
            Stem(channels=self.channels),
            ConvTower(filter_list=filter_list),
            Transformer(
                channels=self.channels,
                num_transformer_layers=self.num_transformer_layers,
            ),
            self.crop_final,
            FinalPointwise(chanels=self.channels),
        ]

        self._heads = {
            head: Sequential([nn.Dense(features=num_channels), SoftPlus()])
            for head, num_channels in heads_channels.items()
        }

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    @nn.compact
    def __call__(self, inputs):
        trunk_embedding = self.trunk(inputs)
        return {
            head: head_module(trunk_embedding)
            for head, head_module in self.heads.items()
        }
