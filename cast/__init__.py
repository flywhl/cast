import importlib
from typing import Any, Callable
from cast.cast import for_type, Cast, CastModel, ValidationContext, RefTagRegistry


def reftag(tag: str):
    """Decorator to register a reference tag handler function."""

    def decorator(handler: Callable):
        RefTagRegistry.register(tag, handler)
        return handler

    return decorator


# Implement built-in reftags
@reftag("import")
def import_tag(path: str, _: ValidationContext) -> Any:
    """Handle @import:module.path.to.thing references."""
    module_path, attr = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(path) from e
    return getattr(module, attr)


@reftag("value")
def value_tag(path: str, context: ValidationContext) -> Any:
    """Handle @value:path.to.value references."""
    return context.get_nested_value(path)


__all__ = ["for_type", "Cast", "CastModel", "reftag"]
