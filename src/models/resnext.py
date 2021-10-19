import torch
import torchvision
from config.value_config import *


PREMODEL = "/home/wujilong/.cache/torch/checkpoints/resnext50_32x4d-7cdf4587.pth"
def make_model(key):
    return ResNext(key)


class ResNext(torch.nn.Module):
    def __init__(self, key):
        super(ResNext, self).__init__()
        backbone = torchvision.models.__dict__[key](pretrained=False)
        dict = torch.load(PREMODEL, map_location="cpu")
        backbone.load_state_dict(dict)
        self.layer0 = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=self.layer4[-1].conv1.in_channels, out_features=8),
        )

        pass

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    from PIL import Image
    path = "/mnt/wfs/wujilong/mlabel_data/ajuke_224x224/ffff0771ea9d6af757a924f291c70ef5_600x600.jpg"
    img = Image.open(path).convert("RGB")
    img = torchvision.transforms.Resize((224, 224))(img)
    img = torchvision.transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    model = make_model("resnext50_32x4d")
    pre = model(img)
    print(torch.nn.functional.softmax(pre, dim=1))
   # print(model.layer3[0].conv1.weight.detach().numpy()[:,:,0,0])