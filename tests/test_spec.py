from dataclasses import dataclass
import random
import statistics
from typing import Sequence, Union, overload

import cast
from cast.spec import Spec, SpecModel


@dataclass
class Tensor:
    """A simple mock tensor class that wraps a list of numbers.

    We use this because we don't want to take torch as a dependency.
    """

    data: list[float]

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "Tensor":
        return cls(list(values))

    def __len__(self) -> int:
        return len(self.data)

    def mean(self) -> float:
        return statistics.mean(self.data)

    def std(self) -> float:
        return statistics.stdev(self.data)

    def __iter__(self):
        return iter(self.data)

    @overload
    def __getitem__(self, idx: int) -> float: ...

    @overload
    def __getitem__(self, idx: slice) -> "Tensor": ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[float, "Tensor"]:
        if isinstance(idx, slice):
            return Tensor(self.data[idx])
        return self.data[idx]


@cast.for_type(Tensor)
class NormalTensor(Spec[Tensor]):
    """Specification for creating tensors from a normal distribution."""

    mean: float
    std_dev: float
    size: int

    def build(self) -> Tensor:
        return Tensor([random.gauss(self.mean, self.std_dev) for _ in range(self.size)])


@cast.for_type(Tensor)
class UniformTensor(Spec[Tensor]):
    """Specification for creating tensors with values from a uniform distribution."""

    low: float
    high: float
    size: int

    def build(self) -> Tensor:
        return Tensor([random.uniform(self.low, self.high) for _ in range(self.size)])


class DataContainer(SpecModel):
    """Example model using spec-enabled tensor."""

    values: Tensor


def test_spec_build():
    """Test the spec building functionality."""
    # Test direct tensor assignment
    direct_tensor = Tensor.from_list([1.0, 2.0, 3.0])
    model1 = DataContainer(values=direct_tensor)
    assert len(model1.values) == 3
    assert list(model1.values) == [1.0, 2.0, 3.0]

    # Test normal distribution spec
    spec_dict = {"values": {"mean": 0.0, "std_dev": 1.0, "size": 1000}}
    model2 = DataContainer.model_validate(spec_dict)
    # Check statistical properties
    assert len(model2.values) == 1000

    # Test validation error for missing required fields
    try:
        DataContainer.model_validate({"values": {"mean": 0.0}})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test uniform distribution spec
    model3 = DataContainer.model_validate(
        {"values": {"low": -1.0, "high": 1.0, "size": 100}}
    )
    assert len(model3.values) == 100
