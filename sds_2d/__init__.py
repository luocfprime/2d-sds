"""
Registry implementation borrowed from https://github.com/threestudio-project/threestudio
"""

__modules__ = {}


def register(name):
    def decorator(cls):
        if name in __modules__:
            raise ValueError(
                f"Module {name} already exists! Names of extensions conflict!"
            )
        else:
            __modules__[name] = cls
        return cls

    return decorator


def instantiate(cfg):
    if cfg.get("name", None) in __modules__:
        return __modules__[cfg.name](cfg)
    raise NotImplementedError(f"{cfg.name} is not implemented yet!")


from . import algorithm, rasterizer, wt_schedule
