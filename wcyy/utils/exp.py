def get_exp_ID(cfg):
    if cfg.get('exp_id', None) is None:
        cfg['exp_id'] = f'exp-{cfg["pretrained_model"]}_e{cfg["epochs"]}_b{cfg["batch_size"]}_{cfg["train_transform"]}_{cfg["valid_transform"]}_explr_{cfg["model"]}_{"freeze" if cfg["freeze"] else "unfreeze"}_{cfg["classes"]}_{cfg["opt_func"]}'
    return cfg['exp_id']