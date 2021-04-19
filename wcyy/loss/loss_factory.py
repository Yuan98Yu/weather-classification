from typing import Dict
import torch.nn.functional as F

from wcyy.loss.focal_loss import FocalLoss


def create_loss_func(cfg: Dict = None):
    if cfg is not None and cfg.get('opt_func', '') == 'focalloss':
        loss_func = FocalLoss(len(cfg['classes']))
    loss_func = F.cross_entropy

    return loss_func
