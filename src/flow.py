from omegaconf import OmegaConf
from metaflow import FlowSpec, Config, Parameter, ConfigValue
from typing import Any

from src.pipelines.plantcad2.evaluation.config import FlowConfig
from src.pipelines.plantcad2.evaluation.common import get_evaluation_config_path


def parse_omegaconf(txt: str) -> dict[str, Any]:
    config = OmegaConf.create(txt)
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def resolve_config(config: ConfigValue, overrides: str) -> FlowConfig:
    cfg = config
    if overrides:
        base_config = OmegaConf.create(cfg.to_dict())
        override_config = OmegaConf.from_dotlist(overrides.split(","))
        merged_config = OmegaConf.merge(base_config, override_config)
        cfg = OmegaConf.to_container(merged_config, resolve=True)
    return FlowConfig.model_validate(cfg)


class BaseFlow(FlowSpec):
    base_config = Config(
        "base_config",
        default=get_evaluation_config_path(),
        parser=parse_omegaconf,
    )

    overrides = Parameter(
        "overrides",
        default="",
        help="Comma-separated list of configuration overrides, e.g. 'paths.data_dir=new_data_dir,tasks.evolutionary_constraint.sample_size=100'",
    )

    def resolve_config(self):
        return resolve_config(self.base_config, self.overrides)
