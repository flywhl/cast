import logging
from contextlib import contextmanager
from threading import local
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ValidationContext:
    """Thread-local storage for validation context."""

    _context = local()

    @classmethod
    @contextmanager
    def root_data(cls, data: dict):
        """Store the root data during validation."""
        # Initialize depth counter if needed
        if not hasattr(cls._context, "depth"):
            cls._context.depth = 0

        # Set data only at top level
        if cls._context.depth == 0 and not hasattr(cls._context, "data"):
            cls._context.data = data

        cls._context.depth += 1
        try:
            yield
        finally:
            cls._context.depth -= 1
            # Only clean up data when unwinding the top level
            if cls._context.depth == 0 and hasattr(cls._context, "data"):
                del cls._context.data

    @classmethod
    def get_root_data(cls) -> Optional[dict]:
        """Get the current root data."""
        return getattr(cls._context, "data", None)

    @classmethod
    def get_nested_value(cls, path: str) -> Any:
        """Get a value from the root data using dot notation path."""
        data = cls.get_root_data()
        if not data:
            raise ValueError(
                f"Cannot get value at {path}, because there is no validation context data."
            )
        keys = path.split(".")
        for key in keys:
            if not isinstance(data, dict):
                raise ValueError(f"Cannot traverse path {path}: {key} is not a dict")
            if key not in data:
                raise ValueError(f"Key {key} not found in path {path}")
            data = data[key]
        return data
