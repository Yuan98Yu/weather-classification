from typing import Dict
from torch import nn
import torchvision.models as models

from wcyy.models.model import ImageClassificationBase
from wcyy.models.resnet import resnet15


class ResFC1CLF(ImageClassificationBase):
    def __init__(self,
                 num_classes=6,
                 pretrained_model='resnet34',
                 pretrained=True,
                 cfg: Dict = None):
        super().__init__(cfg)
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


class ResFC2CLF(ResFC1CLF):
    def __init__(self,
                 num_classes=6,
                 pretrained_model='resnet34',
                 pretrained=True,
                 cfg: Dict = None):
        super.__init__(num_classes, pretrained_model, pretrained, cfg)
        # # Use a pretrained model
        # self.network = getattr(models, pretrained_model)(pretrained=pretrained)

        # # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(
                num_ftrs,
                num_ftrs // 4,
            ), nn.ReLU(), nn.Linear(
                num_ftrs // 4,
                num_ftrs // 2,
            ), nn.ReLU(), nn.Linear(num_ftrs // 2, num_classes))


class Res15FC1CLF(ImageClassificationBase):
    def __init__(self,
                 num_classes,
                 pretrained_model='resnet15',
                 pretrained=False,
                 cfg: Dict = None):
        super().__init__(cfg)
        # Use a pretrained model
        self.pretrained_model = pretrained_model
        self.network = resnet15(pretrained=pretrained,
                                progress=True,
                                num_classes=num_classes)

    def forward(self, xb):
        return self.network(xb)
