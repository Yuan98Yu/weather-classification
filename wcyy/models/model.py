from typing import Dict
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from wcyy.metrics import accuracy
from wcyy.loss import create_loss_func


class ImageClassificationBase(nn.Module):
    # @abc.abstractmethod
    # def forward(self, x):
    #     pass
    def __init__(self, cfg: Dict = None):
        super().__init__()
        self.loss_func = create_loss_func(cfg)

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = self.loss_func(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = self.loss_func(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {
            'y_pred': out,
            'y_true': labels,
            'val_loss': loss.detach(),
            'val_acc': acc
        }

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result, writer: SummaryWriter):
        text = (
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
            .format(epoch, result['train_loss'], result['val_loss'],
                    result['val_acc']))
        print(text)
        writer.add_text('epoch_end_text', text, epoch)
        writer.add_scalar('epoch_train_loss', result['train_loss'], epoch)
        writer.add_scalar('epoch_valid_loss', result['val_loss'], epoch)
        writer.add_scalar('epoch_valid_accuracy', result['val_acc'], epoch)

    # @abc.abstractmethod
    # def freeze(self):
    #     pass

    # @abc.abstractmethod
    # def unfreeze(self):
    #     pass
