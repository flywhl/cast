from typing import TypeVar, Generic, Any, get_type_hints
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, ValidationError
from pydantic_core import core_schema

T = TypeVar("T")


class Spec(BaseModel, Generic[T]):
    """Base class for parameter specifications that can be built into instances."""

    def build(self) -> T:
        raise NotImplementedError

    @property
    def fields(self):
        raise NotImplementedError()


class SpecRegistry:
    """Global registry mapping types to their spec classes."""

    _specs: dict[type, list[type[Spec]]] = {}

    @classmethod
    def register(cls, target_type: type, spec_type: type[Spec]):
        """Register a spec class for a target type."""
        if target_type not in cls._specs:
            cls._specs[target_type] = []
        cls._specs[target_type].append(spec_type)

    @classmethod
    def get_specs(cls, target_type: type) -> list[type[Spec]]:
        """Get all registered specs for a type."""
        return cls._specs.get(target_type, [])


def for_type(target_type: type):
    """Decorator to register a spec class for a given type."""

    def decorator(spec_type: type[Spec]):
        SpecRegistry.register(target_type, spec_type)
        return spec_type

    return decorator


class SpecModel(BaseModel):
    """Base model class that automatically applies spec validation to fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def try_build(cls, target_type: type, value: dict) -> Any:
        """Try each registered spec in order until one works."""
        specs = SpecRegistry.get_specs(target_type)
        if not specs:
            raise ValueError(f"No specs registered for type {target_type}")

        errors = []
        for spec_type in specs:
            try:
                spec = spec_type.model_validate(value)
                return spec.build()
            except (ValidationError, ValueError) as e:
                errors.append(f"{spec_type.__name__}: {str(e)}")
                continue
            except Exception as e:
                # Don't catch unexpected errors
                raise

        error_msg = "\n".join(f"- {err}" for err in errors)
        raise ValueError(
            f"No compatible spec found for {target_type}. Tried:\n{error_msg}"
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        schema = super().__get_pydantic_core_schema__(_source_type, _handler)

        # Get registered types from the model's annotations
        hints = get_type_hints(cls)
        fields_requiring_validation = {
            name: type_
            for name, type_ in hints.items()
            if name in cls.model_fields and SpecRegistry.get_specs(type_)
        }

        if not fields_requiring_validation:
            return schema

        def validate_spec_fields(v: Any) -> Any:
            if not isinstance(v, dict):
                return v

            for field_name, field_type in fields_requiring_validation.items():
                if field_name not in v:
                    continue

                field_value = v[field_name]

                # Skip if value is already of the target type
                if isinstance(field_value, field_type):
                    continue

                if isinstance(field_value, dict):
                    try:
                        v[field_name] = cls.try_build(field_type, field_value)
                    except ValueError as e:
                        raise ValueError(f"Error building {field_name}: {str(e)}")

            return v

        return core_schema.chain_schema(
            [core_schema.no_info_plain_validator_function(validate_spec_fields), schema]
        )
