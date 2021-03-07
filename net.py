import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tensorboardX import SummaryWriter

from data import to_device
from resnet import resnet15


def create_model(cfg, num_classes):
    model = globals()[cfg['model']]
    model = model(num_classes, cfg['pretrained_model'], cfg['pretrained'])
    model = to_device(model, cfg['device'])
    return model


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


class WeartherClassification(ImageClassificationBase):
    def __init__(self,
                 num_classes,
                 pretrained_model='resnet34',
                 pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = getattr(models, pretrained_model)(pretrained=pretrained)

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

    def epoch_end(self, epoch, result, writer: SummaryWriter):
        text = (
            "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
            .format(epoch, result['lrs'][-1], result['train_loss'],
                    result['val_loss'], result['val_acc']))
        print(text)
        writer.add_text('epoch_end_text', text, epoch)
        writer.add_scalar('epoch_train_loss', result['train_loss'], epoch)
        writer.add_scalar('epoch_valid_loss', result['val_loss'], epoch)
        writer.add_scalar('epoch_valid_accuracy', result['val_acc'], epoch)


class WeartherClassification2(ImageClassificationBase):
    def __init__(self,
                 num_classes,
                 pretrained_model='resnet34',
                 pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = getattr(models, pretrained_model)(pretrained=pretrained)

        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return self.network(xb)

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
        writer.add_scalar('epoch_train_loss', result['train_loss'], epoch)
        writer.add_scalar('epoch_valid_loss', result['val_loss'], epoch)
        writer.add_scalar('epoch_valid_accuracy', result['val_acc'], epoch)


class WeatherModel2(WeartherClassification):
    def __init__(self,
                 num_classes=6,
                 pretrained_model='resnet34',
                 pretrained=True):
        super().__init__(num_classes)
        # Use a pretrained model
        self.network = getattr(models, pretrained_model)(pretrained=pretrained)

        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(
                num_ftrs,
                num_ftrs // 4,
            ), nn.ReLU(), nn.Linear(
                num_ftrs // 4,
                num_ftrs // 2,
            ), nn.ReLU(), nn.Linear(num_ftrs // 2, num_classes))

    def epoch_end(self, epoch, result, writer: SummaryWriter):
        text = (
            "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
            .format(epoch, result['lrs'][-1], result['train_loss'],
                    result['val_loss'], result['val_acc']))
        print(text)
        writer.add_text('epoch_end_text', text, epoch)
        writer.add_scalar('epoch_train_loss', result['train_loss'], epoch)
        writer.add_scalar('epoch_valid_loss', result['val_loss'], epoch)
        writer.add_scalar('epoch_valid_accuracy', result['val_acc'], epoch)


class WeatherModel1_2(WeartherClassification2):
    def epoch_end(self, epoch, result, writer: SummaryWriter):
        text = (
            "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
            .format(epoch, result['lrs'][-1], result['train_loss'],
                    result['val_loss'], result['val_acc']))
        print(text)
        writer.add_text('epoch_end_text', text, epoch)
        writer.add_scalar('epoch_train_loss', result['train_loss'], epoch)
        writer.add_scalar('epoch_valid_loss', result['val_loss'], epoch)
        writer.add_scalar('epoch_valid_accuracy', result['val_acc'], epoch)


class WeatherModel3(ImageClassificationBase):
    def __init__(self,
                 num_classes,
                 pretrained_model='resnet34',
                 pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.pretrained_model = pretrained_model
        self.network = timm.create_model(pretrained_model,
                                         pretrained=pretrained)

        # Replace last layer
        if pretrained_model in ['regnetx_016', 'regnetx_032']:
            num_ftrs = self.network.head.fc.in_features
            self.network.head.fc = nn.Linear(num_ftrs, num_classes)
        elif pretrained_model in ['efficientnet_b3a']:
            num_ftrs = self.network.classifier.in_features
            self.network.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False

        if self.pretrained_model in ['regnetx_016', 'regnetx_032']:
            for param in self.network.head.fc.parameters():
                param.require_grad = True
        elif self.pretrained_model in ['efficientnet_b3a']:
            for param in self.network.classifier.parameters():
                param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True


class WeatherModel4(ImageClassificationBase):
    def __init__(self,
                 num_classes,
                 pretrained_model='resnet15',
                 pretrained=False):
        super().__init__()
        # Use a pretrained model
        self.pretrained_model = pretrained_model
        self.network = resnet15(pretrained=pretrained,
                                progress=True,
                                num_classes=num_classes)

    def forward(self, xb):
        return self.network(xb)
