from typing import TypeVar, Generic, Any, get_type_hints, get_args
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, ValidationError
from pydantic_core import core_schema

T = TypeVar("T")


class Cast(BaseModel, Generic[T]):
    """Base class for parameter specifications that can be built into instances."""

    def build(self) -> T:
        raise NotImplementedError

    @property
    def fields(self):
        raise NotImplementedError()


class CastRegistry:
    """Global registry mapping types to their cast classes."""

    _casts: dict[type, list[type[Cast]]] = {}

    @classmethod
    def register(cls, target_type: type, cast_type: type[Cast]):
        """Register a cast class for a target type."""
        if target_type not in cls._casts:
            cls._casts[target_type] = []
        cls._casts[target_type].append(cast_type)

    @classmethod
    def get_casts(cls, target_type: type) -> list[type[Cast]]:
        """Get all registered casts for a type."""
        return cls._casts.get(target_type, [])


def for_type(target_type: type):
    """Decorator to register a cast class for a given type."""

    def decorator(cast_type: type[Cast]):
        CastRegistry.register(target_type, cast_type)
        return cast_type

    return decorator


class CastModel(BaseModel):
    """Base model class that automatically applies cast validation to fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

        # Get registered types from the model's annotations
        hints = get_type_hints(cls)
        print(_source_type)
        print(hints.items())
        fields_requiring_validation = {
            name: type_
            for name, type_ in hints.items()
            if name in cls.model_fields and CastRegistry.get_casts(type_)
        }

        if not fields_requiring_validation:
            return schema

        def validate_cast_fields(v: Any) -> Any:
            # Handle lists directly
            if isinstance(v, list):
                return v

            if not isinstance(v, dict):
                return v

            for field_name, field_type in fields_requiring_validation.items():
                if field_name not in v:
                    continue

                field_value = v[field_name]

                # Skip if value is already of the target type
                if isinstance(field_value, field_type):
                    continue

                # Handle lists of cast types
                if isinstance(field_value, list):
                    list_type = get_args(hints[field_name])[0]
                    if CastRegistry.get_casts(list_type):
                        try:
                            v[field_name] = [
                                cls.try_build(list_type, item)
                                if isinstance(item, dict)
                                else item
                                for item in field_value
                            ]
                        except ValueError as e:
                            raise ValueError(
                                f"Error building item in {field_name}: {str(e)}"
                            )
                    continue

                if isinstance(field_value, dict):
                    try:
                        v[field_name] = cls.try_build(field_type, field_value)
                    except ValueError as e:
                        raise ValueError(f"Error building {field_name}: {str(e)}")

            return v

        return core_schema.chain_schema(
            [core_schema.no_info_plain_validator_function(validate_cast_fields), schema]
        )
