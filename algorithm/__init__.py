from .sds import SDS

algorithm_registry = {
    "sds": SDS,
}


def get_sampling_algorithm(algorithm_cfg):
    if algorithm_cfg.name not in algorithm_registry:
        raise NotImplementedError(f"Algorithm {algorithm_cfg.name} not implemented")
    return algorithm_registry[algorithm_cfg.name](algorithm_cfg)
