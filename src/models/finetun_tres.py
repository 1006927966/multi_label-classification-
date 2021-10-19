import torch.nn as nn
import torch
from config.value_config import CLSNUM, PREMODEL
from src.models.tresnet import TResNet
import argparse




class Tres(nn.Module):
    def __init__(self):
        super(Tres, self).__init__()
        self.classnum = CLSNUM
        self.net = TResNet(layers=[4, 5, 18, 3], num_classes=9605, in_chans=3, width_factor=1.2,
                    do_bottleneck_head=True)
        self.param = torch.load(PREMODEL, map_location="cpu")
        self.net.load_state_dict(self.param["model"])
        self.relu = nn.ReLU()
        self.output = nn.Linear(in_features=9605, out_features=self.classnum)
        # self.body = self.net.body
        # self.global_pool = self.net.global_pool
        # self.fc = nn.Sequential(nn.Linear(in_features=2432, out_features=512, bias=True),
        #                         nn.BatchNorm1d(512),
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(in_features=512, out_features=self.classnum, bias=True))


    def forward(self, x):
        x = self.net(x)
        x = self.relu(x)
        x = self.output(x)
        return x



if __name__ == "__main__":
    # param = torch.load(PREMODEL, map_location="cpu")
    # for key in param["model"].keys():
    #     print(key)
    img = torch.randn(1,3, 224, 224).cuda()
    a = Tres().cuda()
    print(next(a.parameters()).is_cuda)
    print(a(img))
    for name, i in a._modules.items():
        print(name)
    # for param in a.parameters():
    #     print(param)
