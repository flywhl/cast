<p align="center">
  <img src="https://github.com/user-attachments/assets/6004aee7-0a64-4cc4-a7aa-b6790fedccc7">
</p>
<p align="center">
  <b>Use Pydantic to build complex types from parameter specifications.</b>
</p>
<br/>

## Installation

* `uv add git+https://github.com/flywhl/cyantic`

## Features

* Build complex objects using intermediate Pydantic models.
* Reference other values using `@value:x.y.z`
* Import objects using `@import:x.y.z`
* Define custom `@reftag` handlers (see tests)

## Usage

```python
from cyantic import Blueprint, blueprint, CyanticModel
from torch import Tensor
import torch
import yaml

# 1. Create and register some useful parameterisations
#       (or soon install from PyPi, i.e. `rye add cyantic-torch`)

@blueprint(Tensor)
class NormalTensor(Blueprint[Tensor]):

    mean: float
    std: float
    size: tuple[int, ...]

    def build(self) -> Tensor:
        return torch.normal(self.mean, self.std, size=self.size)

@blueprint(Tensor)
class UniformTensor(Blueprint[Tensor]):
    low: float
    high: float
    size: tuple[int, ...]

    def build(self) -> Tensor:
      return torch.empty(self.size).uniform_(self.low, self.high)


# 2. Write pydantic models using `CyanticModel` base class

class MyModel(CyanticModel):
    normal_tensor: Tensor
    uniform_tensor: Tensor


# 3. Validate from YAML files that specify the parameterisation

some_yaml = """common:
    size: [3, 5]
normal_tensor:
    mean: 0.0
    std: 0.1
    size: @value:common.size
uniform_tensor:
    low: -1.0
    std: 1.0
    size: @value:common.size
"""

# 4. Receive objects built from the parameterisations.

my_model = MyModel.model_validate(yaml.safe_load(some_yaml))
assert isinstance(my_model.normal_tensor, Tensor)
assert isinstance(my_model.uniform_tensor, Tensor)
```


## Development

* `git clone https://github.com/flywhl/cyantic.git`
* `cd cyantic`
* `uv sync`

## Flywheel

Science needs humble software tools. [Flywheel](https://flywhl.dev/) is an open source collective building simple tools to preserve scientific momentum, inspired by devtools and devops culture.
