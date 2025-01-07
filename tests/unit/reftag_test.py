from unittest.mock import patch
import os
from pydantic import ValidationError

from cast import CastModel
from cast.context import ValidationContext


class SimpleModel(CastModel):
    """A simple model for testing reftags."""
    name: str


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
