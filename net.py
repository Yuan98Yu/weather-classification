from shutil import Error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tensorboardX import SummaryWriter


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

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
        writer.add_scalar('epoch train loss', result['train_loss'], epoch)
        writer.add_scalar('epoch valid loss', result['val_loss'], epoch)
        writer.add_scalar('epoch valid accuracy', result['val_acc'], epoch)


class WeartherClassification(ImageClassificationBase):
    def __init__(self, num_classes, pretrained_model='resnet34'):
        super().__init__()
        # Use a pretrained model
        self.network = getattr(models, pretrained_model)(pretrained=True)

        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True


class WeatherModel1(WeartherClassification):
    def epoch_end(self, epoch, result, writer: SummaryWriter):
        text = (
            "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
            .format(epoch, result['lrs'][-1], result['train_loss'],
                    result['val_loss'], result['val_acc']))
        print(text)
        writer.add_text('epoch_end_text', text, epoch)
        writer.add_scalar('epoch train loss', result['train_loss'], epoch)
        writer.add_scalar('epoch valid loss', result['val_loss'], epoch)
        writer.add_scalar('epoch valid accuracy', result['val_acc'], epoch)
