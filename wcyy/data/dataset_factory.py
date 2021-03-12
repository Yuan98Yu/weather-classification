from wcyy.data.dataset import CustomDataset
import config


def create_full_dataset(cfg):
    classes = getattr(config, cfg['classes'])
    return CustomDataset(cfg['data_dir'], classes)
