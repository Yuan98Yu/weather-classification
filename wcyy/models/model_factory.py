from wcyy.utils.device import to_device
from wcyy.models.resnet_clf import *
from wcyy.models.timm_clf import *


def create_model(cfg, num_classes):
    model = globals()[cfg['model']]
    model = model(num_classes, cfg['pretrained_model'], cfg['pretrained'], cfg)
    model = to_device(model, cfg['device'])
    return model
