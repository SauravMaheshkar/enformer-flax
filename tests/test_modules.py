import pytest
from flax import linen as nn

from enformer_flax.model import Enformer


@pytest.mark.actions
def test_instance():

    model = Enformer()
    assert isinstance(model, nn.Module)
