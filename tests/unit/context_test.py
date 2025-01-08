from cyantic.context import ValidationContext


def test_validation_context():
    """Test the ValidationContext class functionality."""
    test_data = {"outer": {"inner": {"value": 42}}}

    # Test basic context management
    with ValidationContext.root_data(test_data):
        assert ValidationContext.get_root_data() == test_data
        assert ValidationContext.get_nested_value("outer.inner.value") == 42

    # Test context is cleared after exit
    assert ValidationContext.get_root_data() is None

    # Test nested context management
    with ValidationContext.root_data(test_data):
        with ValidationContext.root_data({"other": "data"}):
            # Inner context should not override outer
            assert ValidationContext.get_root_data() == test_data

    # Test error cases
    try:
        ValidationContext.get_nested_value("nonexistent.path")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        with ValidationContext.root_data({"a": 1}):
            ValidationContext.get_nested_value("a.b.c")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
