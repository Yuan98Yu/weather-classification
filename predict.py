import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from tensorboardX import SummaryWriter

from net import WeatherModel1
from data import DeviceDataLoader, get_default_device, show_batch, to_device
import config


@torch.no_grad()
def predict(model: WeatherModel1, dataloader: DataLoader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in dataloader]
    return model.validation_epoch_end(outputs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, default='res50_e181_explr')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    cfg = getattr(config, args.cfg)
    cfg['ckpt'] = args.ckpt
    cfg['device'] = get_default_device()
    cfg['exp_id'] = f'exp-{args.cfg}'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    exps_root = 'runs'
    exp_id = cfg['exp_id']
    writer = SummaryWriter(os.path.join(exps_root, exp_id))

    # Create datasets
    stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    test_transform = tt.Compose(
        [tt.Resize([200, 200]),
         tt.ToTensor(),
         tt.Normalize(*stats)])
    # set the batch size
    batch_size = 64

    test_ds = ImageFolder(cfg['data_dir'], test_transform)
    classes = test_ds.classes
    print(classes)
    num_classes = len(classes)
    train_ds, valid_ds = torch.utils.data.random_split(test_ds, [50000, 10000])

    test_ds = valid_ds
    train_dl = DeviceDataLoader(
        DataLoader(train_ds, batch_size * 2, num_workers=2, pin_memory=True),
        cfg['device'])
    valid_dl = DeviceDataLoader(
        DataLoader(valid_ds, batch_size * 2, num_workers=2, pin_memory=True),
        cfg['device'])

    show_batch(valid_dl)

    model = to_device(WeatherModel1(num_classes, cfg['pretrained_model']),
                      cfg['device'])
    model_ckpt = torch.load(os.path.join(writer.logdir, cfg['ckpt']))
    model.load_state_dict(model_ckpt)

    # result = predict(model, valid_dl)
    result = predict(model, train_dl)
    print(result)
