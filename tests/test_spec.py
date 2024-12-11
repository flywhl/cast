import random
import statistics
from typing import Sequence, Union, overload

from pydantic import BaseModel, Field

import cast
from cast import Cast, CastModel


class Tensor(BaseModel):
    """A simple mock tensor class that wraps a list of numbers.

    We use this because we don't want to take torch as a dependency.
    """

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


@cast.for_type(Tensor)
class NormalTensor(Cast[Tensor]):
    """Cast for creating tensors from a normal distribution."""

    mean: float
    std_dev: float
    size: int = Field(gt=0, description="Size must be a positive integer")

    def build(self) -> Tensor:
        return Tensor(
            data=[random.gauss(self.mean, self.std_dev) for _ in range(self.size)]
        )


@cast.for_type(Tensor)
class UniformTensor(Cast[Tensor]):
    """Cast for creating tensors with values from a uniform distribution."""

    low: float
    high: float
    size: int = Field(gt=0, description="Size must be a positive integer")

    def build(self) -> Tensor:
        return Tensor(
            data=[random.uniform(self.low, self.high) for _ in range(self.size)]
        )


class DataContainer(CastModel):
    """Example model using cast-enabled tensor."""

    values: Tensor


def test_cast_build():
    """Test the cast building functionality."""
    # Test direct tensor assignment
    direct_tensor = Tensor.from_list([1.0, 2.0, 3.0])
    model1 = DataContainer(values=direct_tensor)
    assert len(model1.values) == 3
    assert list(model1.values.data) == [1.0, 2.0, 3.0]

    # Test normal distribution cast
    cast_dict = {"values": {"mean": 0.0, "std_dev": 1.0, "size": 1000}}
    model2 = DataContainer.model_validate(cast_dict)
    # Check statistical properties
    assert len(model2.values) == 1000

    # Test validation error for missing required fields
    try:
        DataContainer.model_validate({"values": {"mean": 0.0}})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


class TensorList(CastModel):
    """A model containing a list of tensors."""
    tensors: list[Tensor]


def test_list_of_castmodels():
    """Test handling lists of fields that use casts."""
    # Test list of normal distributions
    normal_list_data = {
        "tensors": [
            {"mean": 0.0, "std_dev": 1.0, "size": 10},
            {"mean": 5.0, "std_dev": 2.0, "size": 20},
            {"mean": -5.0, "std_dev": 0.5, "size": 30}
        ]
    }
    
    model = TensorList.model_validate(normal_list_data)
    assert len(model.tensors) == 3
    assert len(model.tensors[0]) == 10
    assert len(model.tensors[1]) == 20
    assert len(model.tensors[2]) == 30
    
    # Verify statistical properties of each tensor
    assert -0.5 < model.tensors[0].mean() < 0.5  # mean ≈ 0.0
    assert 4.5 < model.tensors[1].mean() < 5.5   # mean ≈ 5.0
    assert -5.5 < model.tensors[2].mean() < -4.5 # mean ≈ -5.0
    
    # Test mixed normal and uniform distributions in the same list
    mixed_list_data = {
        "tensors": [
            {"mean": 0.0, "std_dev": 1.0, "size": 100},  # Normal
            {"low": 0.0, "high": 1.0, "size": 100},      # Uniform
            {"mean": 2.0, "std_dev": 0.5, "size": 100}   # Normal
        ]
    }
    
    model = TensorList.model_validate(mixed_list_data)
    assert len(model.tensors) == 3
    assert all(len(t) == 100 for t in model.tensors)
    
    # Check uniform distribution bounds
    assert all(0.0 <= x <= 1.0 for x in model.tensors[1].data)
    
    # Test validation with invalid items in list
    invalid_list_data = {
        "tensors": [
            {"mean": 0.0, "std_dev": 1.0, "size": 10},
            {"mean": 0.0, "size": 20},  # Missing std_dev
            {"low": 0.0, "high": 1.0, "size": 30}
        ]
    }
    
    try:
        TensorList.model_validate(invalid_list_data)
        assert False, "Should have raised ValueError for invalid tensor spec"
    except ValueError:
        pass

    # Test uniform distribution cast
    model3 = DataContainer.model_validate(
        {"values": {"low": -1.0, "high": 1.0, "size": 100}}
    )
    assert len(model3.values) == 100


class NestedConfig(BaseModel):
    """A regular Pydantic model for configuration."""
    name: str
    scale: float


class NestedTensorContainer(CastModel):
    """A CastModel that will be nested inside another CastModel."""
    config: NestedConfig
    tensor: Tensor


class ComplexDataContainer(CastModel):
    """A CastModel containing both regular fields and nested CastModels."""
    name: str
    primary: Tensor
    secondary: NestedTensorContainer


def test_nested_cast_models():
    """Test nested CastModels with mixed BaseModel types."""
    # Test nested structure with both normal and cast fields
    nested_data = {
        "name": "test_complex",
        "primary": {
            "mean": 0.0,
            "std_dev": 1.0,
            "size": 50
        },
        "secondary": {
            "config": {
                "name": "nested",
                "scale": 2.0
            },
            "tensor": {
                "low": -1.0,
                "high": 1.0,
                "size": 30
            }
        }
    }
    
    model = ComplexDataContainer.model_validate(nested_data)
    
    # Verify top level fields
    assert model.name == "test_complex"
    assert len(model.primary) == 50
    
    # Verify nested structure
    assert model.secondary.config.name == "nested"
    assert model.secondary.config.scale == 2.0
    assert len(model.secondary.tensor) == 30


def test_mixed_model_validation():
    """Test validation behavior with mixed model types."""
    # Test validation with missing nested fields
    invalid_data = {
        "name": "test_invalid",
        "primary": {"mean": 0.0, "std_dev": 1.0, "size": 50},
        "secondary": {
            "config": {"name": "nested"},  # Missing scale
            "tensor": {"low": -1.0, "high": 1.0, "size": 30}
        }
    }
    
    try:
        ComplexDataContainer.model_validate(invalid_data)
        assert False, "Should have raised ValueError for missing scale"
    except ValueError:
        pass

    # Test validation with invalid cast parameters
    invalid_cast_data = {
        "name": "test_invalid",
        "primary": {"mean": 0.0, "std_dev": 1.0, "size": -50},  # Invalid size
        "secondary": {
            "config": {"name": "nested", "scale": 2.0},
            "tensor": {"low": -1.0, "high": 1.0, "size": 30}
        }
    }
    
    try:
        ComplexDataContainer.model_validate(invalid_cast_data)
        assert False, "Should have raised ValueError for negative size"
    except ValueError:
        pass


def test_cast_type_inference():
    """Test that the cast system correctly infers and applies different cast types."""
    # Test that both NormalTensor and UniformTensor casts work in the same model
    mixed_data = {
        "name": "test_mixed",
        "primary": {
            "mean": 0.0,
            "std_dev": 1.0,
            "size": 100
        },
        "secondary": {
            "config": {
                "name": "nested",
                "scale": 2.0
            },
            "tensor": {
                "low": 0.0,
                "high": 1.0,
                "size": 100
            }
        }
    }
    
    model = ComplexDataContainer.model_validate(mixed_data)
    
    # Verify that different cast types were correctly applied
    assert len(model.primary) == 100
    assert len(model.secondary.tensor) == 100
    
    # Statistical properties check for normal distribution
    assert -0.5 < model.primary.mean() < 0.5  # Should be close to 0.0
    assert 0.8 < model.primary.std() < 1.2    # Should be close to 1.0
    
    # Range check for uniform distribution
    assert all(0.0 <= x <= 1.0 for x in model.secondary.tensor.data)


class TensorWrapper(CastModel):
    """A simple CastModel that will be nested in a BaseModel."""
    tensor: Tensor


class ConfigWithTensor(BaseModel):
    """A BaseModel that contains a CastModel."""
    name: str
    description: str | None = None
    data: TensorWrapper


def test_castmodel_in_basemodel():
    """Test a CastModel nested inside a regular Pydantic BaseModel."""
    config_data = {
        "name": "test_config",
        "description": "Testing CastModel inside BaseModel",
        "data": {
            "tensor": {
                "mean": 0.0,
                "std_dev": 1.0,
                "size": 50
            }
        }
    }
    
    model = ConfigWithTensor.model_validate(config_data)
    
    # Verify BaseModel fields
    assert model.name == "test_config"
    assert model.description == "Testing CastModel inside BaseModel"
    
    # Verify the nested CastModel was properly processed
    assert len(model.data.tensor) == 50
    assert -0.5 < model.data.tensor.mean() < 0.5  # Should be close to 0.0
    assert 0.8 < model.data.tensor.std() < 1.2    # Should be close to 1.0
    
    # Test validation with invalid nested cast
    invalid_data = {
        "name": "test_invalid",
        "data": {
            "tensor": {
                "mean": 0.0,
                "std_dev": 1.0,
                "size": -10  # Invalid negative size
            }
        }
    }
    
    try:
        ConfigWithTensor.model_validate(invalid_data)
        assert False, "Should have raised ValueError for negative size"
    except ValueError:
        pass
