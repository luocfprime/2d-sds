"""
Credits to threestudio
"""

from dataclasses import dataclass

from omegaconf import OmegaConf

from .typings import Any, DictConfig, Optional, Union


def to_primitive(cfg, resolve=True):
    # convert omegaconf to primitive types, avoid errors in calls of type(cfg) and isinstance(cfg, ...)
    if (
        isinstance(cfg, float)
        or isinstance(cfg, int)
        or isinstance(cfg, str)
        or isinstance(cfg, list)
        or isinstance(cfg, dict)
    ):
        return cfg
    return OmegaConf.to_container(cfg, resolve=resolve)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))

    return scfg


class ValidateConfigMixin:
    @dataclass
    class Config:
        pass

    cfg: Config

    def validate_config(self, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
        return parse_structured(self.Config, cfg)
