
from torchvision.models import resnet50
import torch.nn as nn

model = resnet50(pretrained=True)

model.fc = nn.Linear(512, 2)
