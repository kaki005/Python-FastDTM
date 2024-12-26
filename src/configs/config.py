from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ModelConfig:
    name: str = "bert"
    num_topic: int = 0
    SGLD_a: float = 0.1
    SGLD_b: float = 0.1
    SGLD_c: float = 0.1
    phi_var: float = 0.1
    eta_var: float = 0.1
    alpha_var: float = 0.1
    seed: int = 10


@dataclass_json
@dataclass
class DataConfig:
    epochs: int = 10
    # data_dir: str = ""
    output_dir: str = ""


# @dataclass_json
# @dataclass
# class WandbConfig:
#     entity: str = ""
#     project: str = ""


@dataclass_json
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    # wandb: WandbConfig = field(default_factory=WandbConfig)
