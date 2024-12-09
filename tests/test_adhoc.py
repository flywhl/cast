from typing import List
import random
import statistics

import cast
from cast.spec import Spec, SpecModel


@cast.for_type(List)
class NormalList(Spec[List[float]]):
    """Specification for creating lists of numbers from a normal distribution."""

    mean: float
    std_dev: float
    size: int

    def build(self) -> List[float]:
        return [random.gauss(self.mean, self.std_dev) for _ in range(self.size)]


@cast.for_type(List)
class UniformList(Spec[List[float]]):
    """Specification for creating lists with values from a uniform distribution."""

    low: float = 0.0
    high: float = 1.0
    size: int = 10

    def build(self) -> List[float]:
        return [random.uniform(self.low, self.high) for _ in range(self.size)]


class DataContainer(SpecModel):
    """Example model using spec-enabled list."""

    values: List[float]


def test_spec_build():
    """Test the spec building functionality."""
    # Test direct list assignment
    direct_list = [1.0, 2.0, 3.0]
    model1 = DataContainer(values=direct_list)
    assert model1.values == direct_list

    # Test normal distribution spec
    spec_dict = {"values": {"mean": 0.0, "std_dev": 1.0, "size": 1000}}
    model2 = DataContainer.model_validate(spec_dict)
    # Check statistical properties
    assert len(model2.values) == 1000
    assert -0.5 < statistics.mean(model2.values) < 0.5  # roughly zero mean
    assert 0.5 < statistics.stdev(model2.values) < 1.5  # roughly unit std dev

    # Test validation error
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
    assert all(-1.0 <= x <= 1.0 for x in model3.values)
