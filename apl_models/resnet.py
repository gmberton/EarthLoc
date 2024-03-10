
import torch.nn as nn
import torchvision

class ResNet(nn.Module):
    def __init__(self,
                 model_name='resnet50',
                 layers_to_freeze=2,
                 layers_to_crop=[4],
                 ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture. 
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        weights = 'IMAGENET1K_V1'

        if 'resnet50' in model_name:
            self.model = torchvision.models.resnet50(weights=weights)
        elif '101' in model_name:
            self.model = torchvision.models.resnet101(weights=weights)
        elif '152' in model_name:
            self.model = torchvision.models.resnet152(weights=weights)
        elif '34' in model_name:
            self.model = torchvision.models.resnet34(weights=weights)
        elif '18' in model_name:
            self.model = torchvision.models.resnet18(weights=weights)

        if layers_to_freeze >= 0:
            self.model.conv1.requires_grad_(False)
            self.model.bn1.requires_grad_(False)
        if layers_to_freeze >= 1:
            self.model.layer1.requires_grad_(False)
        if layers_to_freeze >= 2:
            self.model.layer2.requires_grad_(False)
        if layers_to_freeze >= 3:
            self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if '34' in model_name or '18' in model_name:
            out_channels = 512
            
        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x

