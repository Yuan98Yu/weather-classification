import timm
from torch import nn

from wcyy.models.model import ImageClassificationBase


class TimmFC1CLF(ImageClassificationBase):
    def __init__(self,
                 num_classes,
                 pretrained_model='efficientnet_b3a',
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


class TimmFC3CLF(TimmFC1CLF):
    def __init__(self,
                 num_classes,
                 pretrained_model='efficientnet_b3a',
                 pretrained=True):
        super().__init__(num_classes, pretrained_model, pretrained)
        # # Use a pretrained model
        # self.pretrained_model = pretrained_model
        # self.network = timm.create_model(pretrained_model,
        #                                  pretrained=pretrained)
        # Replace last layer
        if pretrained_model in ['regnetx_016', 'regnetx_032']:
            num_ftrs = self.network.head.fc.in_features
            self.network.head.fc = nn.Sequential(
                nn.Linear(in_features=num_ftrs, out_features=num_ftrs // 2),
                nn.ReLU(), nn.Dropout(p=0.5),
                nn.Linear(in_features=num_ftrs // 2,
                          out_features=num_ftrs // 4),
                nn.ReLU(), nn.Dropout(p=0.5),
                nn.Linear(num_ftrs // 4, num_classes))
        elif pretrained_model in ['efficientnet_b3a']:
            num_ftrs = self.network.classifier.in_features
            self.network.classifier = nn.Sequential(
                nn.Linear(in_features=num_ftrs, out_features=num_ftrs // 2),
                nn.ReLU(), nn.Dropout(p=0.5),
                nn.Linear(in_features=num_ftrs // 2,
                          out_features=num_ftrs // 4),
                nn.ReLU(), nn.Dropout(p=0.5),
                nn.Linear(num_ftrs // 4, num_classes))
