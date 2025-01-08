import os
import random
import statistics
from typing import Sequence, Union, overload
from pydantic import BaseModel, Field, ValidationError

from cyantic import Blueprint, blueprint, CyanticModel


class SimpleModel(CyanticModel):
    """A simple model for testing hooks."""

    name: str


class Tensor(BaseModel):
    """A simple mock tensor class that wraps a list of numbers."""

    data: list[float]

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "Tensor":
        return cls(data=list(values))

    def __len__(self) -> int:
        return len(self.data)

    def mean(self) -> float:
        return statistics.mean(self.data)

    def std(self) -> float:
        return statistics.stdev(self.data)

    @overload
    def __getitem__(self, idx: int) -> float: ...

    @overload
    def __getitem__(self, idx: slice) -> "Tensor": ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[float, "Tensor"]:
        if isinstance(idx, slice):
            return Tensor(data=self.data[idx])
        return self.data[idx]


@blueprint(Tensor)
class NormalTensor(Blueprint[Tensor]):
    """Blueprint for creating tensors from a normal distribution."""

    mean: float
    std_dev: float
    size: int = Field(gt=0, description="Size must be a positive integer")

    def build(self) -> Tensor:
        return Tensor(
            data=[random.gauss(self.mean, self.std_dev) for _ in range(self.size)]
        )


@blueprint(Tensor)
class UniformTensor(Blueprint[Tensor]):
    """Blueprint for creating tensors with values from a uniform distribution."""

    low: float
    high: float
    size: int = Field(gt=0, description="Size must be a positive integer")

    def build(self) -> Tensor:
        return Tensor(
            data=[random.uniform(self.low, self.high) for _ in range(self.size)]
        )


class DataContainer(CyanticModel):
    """Example model using blueprint-enabled tensor."""

    values: Tensor


class NestedConfig(BaseModel):
    """A regular Pydantic model for configuration."""

    name: str
    scale: float


class NestedTensorContainer(CyanticModel):
    """A CyanticModel that will be nested inside another CyanticModel."""

    config: NestedConfig
    tensor: Tensor


class ComplexDataContainer(CyanticModel):
    """A CyanticModel containing both regular fields and nested CyanticModels."""

    name: str
    primary: Tensor
    secondary: NestedTensorContainer


def test_value_reference():
    """Test the @value reference functionality."""
    # Test basic value reference
    data = {
        "stuff": {"tensor": {"mean": 0.0, "std_dev": 1.0, "size": 50}},
        "values": "@value:stuff.tensor",
    }

    model = DataContainer.model_validate(data)
    assert isinstance(model.values, Tensor)
    assert len(model.values) == 50

    # Test nested value reference
    nested_data = {
        "config": {
            "tensors": {
                "primary": {"mean": 0.0, "std_dev": 1.0, "size": 30},
                "secondary": {"low": -1.0, "high": 1.0, "size": 20},
            }
        },
        "name": "foo",
        "primary": "@value:config.tensors.primary",
        "secondary": {
            "config": {"name": "nested", "scale": 2.0},
            "tensor": "@value:config.tensors.secondary",
        },
    }

    model = ComplexDataContainer.model_validate(nested_data)
    assert len(model.primary) == 30
    assert len(model.secondary.tensor) == 20

    # Test invalid path
    invalid_data = {"values": "@value:nonexistent.path"}
    try:
        DataContainer.model_validate(invalid_data)
        assert False, "Should have raised ValueError for invalid path"
    except ValueError:
        pass

    # Test invalid traversal
    invalid_traverse = {
        "nested": {"value": 123},
        "values": "@value:nested.value.deeper",
    }
    try:
        DataContainer.model_validate(invalid_traverse)
        assert False, "Should have raised ValueError for invalid traversal"
    except ValueError:
        pass


def test_import_reference(mocker):
    """Test the @import reference functionality."""
    # Mock module with test object
    mock_module = mocker.Mock()
    mock_module.test_value = Tensor.from_list([1.0, 2.0, 3.0])

    # Setup mock for importlib.import_module
    mock_import = mocker.patch("importlib.import_module")

    def _side_effect(name: str):
        if name == "test.module":
            return mock_module
        raise ImportError(f"No module named '{name}'")

    mock_import.side_effect = _side_effect

    # Test successful import
    data = {"values": "@import:test.module.test_value"}
    model = DataContainer.model_validate(data)
    assert isinstance(model.values, Tensor)
    assert model.values.data == [1.0, 2.0, 3.0]

    # Test invalid module
    data = {"values": "@import:nonexistent.module.value"}
    try:
        DataContainer.model_validate(data)
        assert False, "Should have raised ValueError for invalid import"
    except ValueError as e:
        pass

    # Test invalid reference format
    data = {"values": "@invalid:something"}
    try:
        DataContainer.model_validate(data)
        assert False, "Should have raised ValueError for invalid reference type"
    except ValueError as e:
        pass


def test_env_reference():
    """Test the @env reference functionality."""
    # Test successful env var reference
    os.environ["TEST_VAR"] = "test_value"
    data = {"name": "@env:TEST_VAR"}
    model = SimpleModel.model_validate(data)
    assert model.name == "test_value"

    # Test missing env var raises ValidationError
    data = {"name": "@env:NONEXISTENT_VAR"}
    try:
        SimpleModel.model_validate(data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Environment variable NONEXISTENT_VAR not found" in str(e)
