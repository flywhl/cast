import importlib
import logging
import os
from typing import Any, Callable

from .context import ValidationContext

logger = logging.getLogger(__name__)

REFTAG_PREFIX = "@"


class RefTagRegistry:
    """Global registry mapping reference tags to their handler functions."""

    _handlers: dict[str, Callable] = {}

    @classmethod
    def register(cls, tag: str, handler: Callable):
        """Register a handler function for a reference tag."""
        if tag in cls._handlers:
            raise ValueError(f"Handler already registered for tag: {tag}")
        cls._handlers[tag] = handler
        logger.debug(f"Registered handler for tag: {tag}")

    @classmethod
    def get_handler(cls, tag: str) -> Callable:
        """Get the handler for a reference tag."""
        if tag not in cls._handlers:
            raise ValueError(f"No handler registered for tag: {tag}")
        return cls._handlers[tag]

    @classmethod
    def clear(cls):
        """Clear all registered handlers."""
        cls._handlers.clear()


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


@reftag("env")
def env_tag(name: str, _: ValidationContext) -> str:
    """Handle @env:VARIABLE_NAME references."""
    try:
        return os.environ[name]
    except KeyError as e:
        raise ValueError(f"Environment variable {name} not found") from e
