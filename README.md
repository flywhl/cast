<p align="center">
  <img src="https://github.com/user-attachments/assets/ff3422a1-9f4a-4c10-bbae-9d55a295938d">
</p>
<p align="center">
  <b>Use Pydantic to build complex types from parameter specifications.</b>
</p>
<br/>

## Note

_Cast_ is still proof-of-concept. Please try the `main` branch, but don't use the library for production work.

## Installation

* `rye add cast --git https://github.com/flywhl/cast`

## Usage

```python
import cast
from cast import Cast, CastModel
from torch import Tensor
import yaml

# 1. Create and register some useful parameterisations
#       (or soon install from PyPi, i.e. `rye add cast-torch`)

@cast.for_type(Tensor)
class NormalTensor(Cast[Tensor]):

    mean: float
    std: float
    size: tuple[int, ...]

    def build(self) -> Tensor:
        ...

@cast.for_type(Tensor)
class UniformTensor(Cast[Tensor]):
    low: float
    high: float
    size: tuple[int, ...]

    def build(self) -> Tensor:
      ...


# 2. Write pydantic models using `CastModel` base class

class MyModel(CastModel):
    normal_tensor: Tensor
    uniform_tensor: Tensor


# 3. Validate from YAML files that specify the parameterisation

some_yaml = """normal_tensor:
    mean: 0.0
    std: 0.1
    size: [3, 5]
uniform_tensor:
    low: -1.0
    std: 1.0
    size: [3, 5]
"""

# 4. Receive objects built from the parameterisations.

my_model = MyModel.model_validate(yaml.safe_load(some_yaml))
assert isinstance(my_model.normal_tensor, Tensor)
assert isinstance(my_model.uniform_tensor, Tensor)
```


## Development

* `git clone https://github.com/flywhl/cast.git`
* `cd cast`
* `rye sync`

## Flywheel

Science needs humble software tools. [Flywheel](https://flywhl.dev/) is an open source collective building simple tools to preserve scientific momentum, inspired by devtools and devops culture.
