
import math
import torch
import torchvision.transforms as tfm
import torch.nn.functional as F

from apl_models import mixvpr
from apl_models import resnet


class APLModel(torch.nn.Module):
    def __init__(self, image_size=320, desc_dim=4096):
        super().__init__()
        self.image_size = image_size
        self.backbone = resnet.ResNet()
        out_channels = desc_dim // 4
        self.aggregator = mixvpr.MixVPR(
            in_channels=1024,
            in_h=math.ceil(image_size/16),
            in_w=math.ceil(image_size/16),
            out_channels=out_channels,
            mix_depth=4,
            mlp_ratio=1,
            out_rows=4
        )
        self.fc = torch.nn.Linear(desc_dim, desc_dim)
        self.desc_dim = desc_dim  # Dimension of final descriptor
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1)
        return x
