import importlib
from contextlib import contextmanager
from threading import local
from typing import Optional, TypeVar, Generic, Any, get_type_hints, get_args
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, ValidationError
from pydantic_core import core_schema

T = TypeVar("T")


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


class CastRegistry:
    """Global registry mapping types to their cast classes."""

    _casts: dict[type, list[type["Cast"]]] = {}

    @classmethod
    def register(cls, target_type: type, cast_type: type["Cast"]):
        """Register a cast class for a target type."""
        if target_type not in cls._casts:
            cls._casts[target_type] = []
        cls._casts[target_type].append(cast_type)

    @classmethod
    def get_casts(cls, target_type: type) -> list[type["Cast"]]:
        """Get all registered casts for a type."""
        return cls._casts.get(target_type, [])


def for_type(target_type: type):
    """Decorator to register a cast class for a given type."""

    def decorator(cast_type: type["Cast"]):
        CastRegistry.register(target_type, cast_type)
        return cast_type

    return decorator


class CastModel(BaseModel):
    """Base model class that automatically applies cast validation to fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _get_fields_requiring_validation(hints: dict[str, Any]) -> set[str]:
        """Get fields that need cast validation."""
        return {
            name
            for name, type_ in hints.items()
            if (
                CastRegistry.get_casts(type_)
                or (
                    hasattr(type_, "__origin__")
                    and type_.__origin__ is list
                    and CastRegistry.get_casts(get_args(type_)[0])
                )
            )
        }

    @staticmethod
    def _validate_list_field(
        field_name: str, field_value: list, list_type: type
    ) -> list:
        """Validate and build a list field."""
        try:
            return [
                CastModel.try_build(list_type, item) if isinstance(item, dict) else item
                for item in field_value
            ]
        except ValueError as e:
            raise ValueError(f"Error building item in {field_name}: {str(e)}")

    @staticmethod
    def _get_raw_type(field_type: type) -> type:
        """Get the raw type without generic parameters."""
        return (
            field_type.__origin__ if hasattr(field_type, "__origin__") else field_type
        )

    @classmethod
    def _process_import_reference(cls, path: str) -> Any:
        module_path, attr = path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ValueError(path) from e
        return getattr(module, attr)

    @classmethod
    def _process_reference(cls, reference: str) -> Any:
        assert reference.startswith("@")
        ref_type, ref_value = reference[1:].split(":")  # slice out the @-prefix
        if ref_type == "import":
            return cls._process_import_reference(ref_value)
        elif ref_type == "value":
            return ValidationContext.get_nested_value(ref_value)
        else:
            raise ValueError(f"Unknown reference type: {ref_type}")

    @classmethod
    def validate_cast_fields(
        cls, v: Any, fields_requiring_validation: set[str], hints: dict[str, Any]
    ) -> Any:
        """Validate and build cast fields in a model."""
        if not isinstance(v, dict):
            return v

        # Reference Loop
        for field_name, field_value in v.items():
            if field_name not in hints:
                continue
            requires_resolution = (
                isinstance(field_value, str)
                and field_value.startswith("@")  # @todo: make this more robust
            )

            if requires_resolution:
                v[field_name] = cls._process_reference(field_value)

        # Cast Loop
        for field_name, field_value in v.items():
            if field_name not in hints:
                continue
            field_type = hints[field_name]
            requires_cast = field_name in fields_requiring_validation

            if requires_cast:
                # Handle lists of cast types first
                if isinstance(field_value, list):
                    list_type = get_args(hints[field_name])[0]
                    if CastRegistry.get_casts(list_type):
                        v[field_name] = CastModel._validate_list_field(
                            field_name, field_value, list_type
                        )
                    continue

                # For non-list fields, skip if value is already of the target type
                raw_type = CastModel._get_raw_type(field_type)
                if isinstance(field_value, raw_type):
                    continue

                if isinstance(field_value, dict):
                    try:
                        v[field_name] = CastModel.try_build(field_type, field_value)
                    except ValueError as e:
                        raise ValueError(f"Error building {field_name}: {str(e)}")

        return v

    @classmethod
    def try_build(cls, target_type: type, value: dict) -> Any:
        """Try each registered cast in order until one works."""
        casts = CastRegistry.get_casts(target_type)
        if not casts:
            raise ValueError(f"No casts registered for type {target_type}")

        errors = []
        for cast_type in casts:
            try:
                cast = cast_type.model_validate(value)
                return cast.build()
            except (ValidationError, ValueError) as e:
                errors.append(f"{cast_type.__name__}: {str(e)}")
                continue
            except Exception as e:
                # Don't catch unexpected errors
                raise

        error_msg = "\n".join(f"- {err}" for err in errors)
        raise ValueError(
            f"No compatible cast found for {target_type}. Tried:\n{error_msg}"
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        schema = super().__get_pydantic_core_schema__(_source_type, _handler)

        hints = get_type_hints(cls)
        fields_requiring_validation = cls._get_fields_requiring_validation(hints)

        if not fields_requiring_validation:
            return schema

        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(
                    lambda v: cls.validate_cast_fields(
                        v, fields_requiring_validation, hints
                    )
                ),
                schema,
            ]
        )

    @classmethod
    def model_validate(cls, obj: Any, *args, **kwargs):
        """Validate and build cast fields in a model."""
        if not isinstance(obj, dict):
            return super().model_validate(obj, *args, **kwargs)

        with ValidationContext.root_data(obj):
            return super().model_validate(obj, *args, **kwargs)


class Cast(CastModel, Generic[T]):
    """Base class for parameter specifications that can be built into instances."""

    def build(self) -> T:
        raise NotImplementedError

    @property
    def fields(self):
        raise NotImplementedError()
