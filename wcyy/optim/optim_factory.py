import torch
import adabound


def create_optimizer(cfg):
    opt_func_name = cfg.get('opt_func', None)
    if opt_func_name == 'adabound':
        opt_func = adabound.AdaBound
    elif opt_func_name is not None:
        opt_func = getattr(torch.optim, opt_func_name)
    else:
        opt_func = torch.optim.Adam

    return opt_func
