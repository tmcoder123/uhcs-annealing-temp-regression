import torch
import torch.nn as nn
import torchvision.models as models

def build_resnet18_regressor():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 1)
    )
    return model

class ResNetLatentExtractor(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.backbone = nn.Sequential(*list(full_model.children())[:-1])
        self.proj = full_model.fc[0] # Linear(512, 64)
    def forward(self, x):
        feat = torch.flatten(self.backbone(x), 1)
        return self.proj(feat)
