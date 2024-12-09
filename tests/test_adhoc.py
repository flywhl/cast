from torch import Tensor
import torch

import cast
from cast.spec import Spec, SpecModel


@cast.for_type(Tensor)
class NormalTensor(Spec[Tensor]):
    """Specification for creating weight tensors."""

    mean: float
    var: float
    shape: tuple[int, ...]

    def build(self) -> Tensor:
        return torch.normal(self.mean, self.var, size=self.shape)


@cast.for_type(Tensor)
class UniformTensor(Spec[Tensor]):
    """Specification for creating tensors with values from a uniform distribution."""

    low: float = 0.0
    high: float = 1.0
    shape: tuple[int, ...]

    def build(self) -> Tensor:
        return torch.empty(self.shape).uniform_(self.low, self.high)


class MyNetwork(SpecModel):
    """Example model using spec-enabled tensor."""

    weights: Tensor


def test_spec_build():
    """Test the spec building functionality."""
    # Test direct tensor assignment
    foo = torch.as_tensor([1, 2, 3])
    model1 = MyNetwork(weights=foo)
    print(model1)

    # Test spec-based construction
    spec_dict = {"weights": {"mean": 0.0, "var": 0.1, "shape": [2, 3, 5]}}
    model2 = MyNetwork.model_validate(spec_dict)
    print(model2)

    try:
        MyNetwork.model_validate({"weights": {"mean": 0.0}})
    except ValueError:
        pass

    model3 = MyNetwork.model_validate(
        {"weights": {"low": -1.0, "high": 1.0, "shape": (3, 4)}}
    )
    print(model3)
