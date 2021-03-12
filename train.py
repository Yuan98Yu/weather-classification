import os
from argparse import ArgumentParser
from typing import Dict
from copy import copy

import nni
# from nni.utils import merge_parameter
import torch
from torch import nn
from torch.utils.data import DataLoader
# import torchvision.transforms as tt
from tensorboardX import SummaryWriter
import numpy as np

from wcyy.data import DeviceDataLoader, create_full_dataset
from wcyy.utils.device import get_default_device
from wcyy.utils.exp import get_exp_ID
from wcyy.models import create_model
from wcyy.optim import create_optimizer
import config


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(writer,
                  start_epoch,
                  epochs,
                  max_lr,
                  model,
                  train_loader,
                  val_loader,
                  weight_decay=0,
                  grad_clip=None,
                  opt_func=torch.optim.SGD,
                  accumulation_steps=16):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    # sched = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    last_epoch = -1 if start_epoch == 0 else start_epoch
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                   gamma=0.95,
                                                   last_epoch=last_epoch)

    for epoch in range(start_epoch, epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for i, batch in enumerate(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            if i % accumulation_steps == 0 or i == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

        # Record & update learning rate
        lrs.append(get_lr(optimizer))
        sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        nni.report_intermediate_result(result['val_acc'])
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result, writer)
        history.append(result)
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(writer.logdir, f'model_epoch{epoch}.ckpt'))
    nni.report_final_result(history[-1]['val_acc'])
    return history


def main(writer: SummaryWriter, cfg: Dict):
    # some data transforms and augmentation to improve accuracy
    # stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # set the batch size
    batch_size = cfg['batch_size']

    train_transform = getattr(config, cfg['train_transform'])
    valid_transform = getattr(config, cfg['valid_transform'])

    full_dataset = create_full_dataset(cfg)
    print(full_dataset.classes)
    num_classes = len(full_dataset.classes)
    print(len(full_dataset))
    # classes = full_dataset.classes
    train_ds, valid_ds = torch.utils.data.random_split(full_dataset,
                                                       [len(full_dataset)-10000, 10000])
    train_ds.dataset = copy(full_dataset)
    train_ds.dataset.transform = train_transform
    valid_ds.dataset.transform = valid_transform
    # test_ds = valid_ds
    train_dl = DeviceDataLoader(
        DataLoader(train_ds,
                   batch_size,
                   shuffle=True,
                   num_workers=3,
                   pin_memory=True), cfg['device'])
    valid_dl = DeviceDataLoader(
        DataLoader(valid_ds, batch_size, num_workers=2, pin_memory=True),
        cfg['device'])

    # show_batch(train_dl)

    # model
    start_epoch = 0
    model = create_model(cfg, num_classes)
    if cfg['freeze']:
        model.freeze()
    if cfg['ckpt']:
        model_ckpt = torch.load(os.path.join(writer.logdir, cfg['ckpt']))
        model.load_state_dict(model_ckpt)
        import re
        start_epoch = int(re.search(r'epoch(\d+)',
                                    cfg['ckpt']).group(0)[5:]) + 1

    # train & val
    epochs = cfg['epochs']
    max_lr = cfg['max_lr']
    grad_clip = cfg['grad_clip']
    weight_decay = cfg['weight_decay']
    opt_func = create_optimizer(cfg)

    history = [evaluate(model, valid_dl)]

    history += fit_one_cycle(writer,
                             start_epoch,
                             epochs,
                             max_lr,
                             model,
                             train_dl,
                             valid_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)

    # plot accuracy
    # accuracies = [x['val_acc'] for x in history]
    # plt.plot(accuracies, '-x')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.title('Accuracy vs. No. of epochs')
    # plt.show()

    # # plot losses
    # train_losses = [x.get('train_loss') for x in history]
    # val_losses = [x['val_loss'] for x in history]
    # plt.plot(train_losses, '-bx')
    # plt.plot(val_losses, '-rx')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['Training', 'Validation'])
    # plt.title('Loss vs. No. of epochs')
    # plt.show()

    # # plor learning rates
    # lrs = np.concatenate([x.get('lrs', []) for x in history])
    # plt.plot(lrs)
    # plt.xlabel('Batch no.')
    # plt.ylabel('Learning rate')
    # plt.title('Learning Rate vs. Batch no.')
    # plt.show()

    torch.save(model.state_dict(), os.path.join(writer.logdir, 'model.ckpt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    cfg = getattr(config, args.cfg)
    try:
        tuner_params = nni.get_next_parameter()
        cfg.update(tuner_params)
        cfg['device'] = get_default_device()
        get_exp_ID(cfg)
        cfg['ckpt'] = args.ckpt

        exps_root = 'runs'
        exp_id = cfg['exp_id']
        writer = SummaryWriter(os.path.join(exps_root, exp_id))

        torch.save(cfg, os.path.join(writer.logdir, 'cfg'))

        main(writer, cfg)
    except Exception as exception:
        print(exception)
        raise
