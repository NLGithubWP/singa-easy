
from torchvision.models import resnet50

model = resnet50(pretrained=True)

model.add(nn.Linear(1000, 2))
