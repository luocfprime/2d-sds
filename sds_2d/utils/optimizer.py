import torch


def get_optimizer(optimizer_cfg, parameters):
    opt_cls = getattr(torch.optim, optimizer_cfg.name)

    if optimizer_cfg.opt_args is None:
        opt_args = {}
    else:
        opt_args = optimizer_cfg.opt_args

    return opt_cls(parameters, **opt_args)
