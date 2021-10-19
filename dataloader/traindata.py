from torch.utils import data
import torchvision
import os
from PIL import Image
import cv2
import torch

class Traindata(data.Dataset):
    def __init__(self, txtpath, nclass):
        self.nclass = nclass
        self.txtpath = txtpath
        with open(txtpath, "r") as f:
            lines = f.readlines()
        self.picpaths = []
        self.labels = []
        for line in lines:
            line = line.strip()
            factors = line.split(',')
            self.picpaths.append(factors[0])
            label = []
            for j in range(1, len(factors)):
                label.append(int(factors[j]))
            self.labels.append(label)
        #self.picpaths = self.picpaths[:100]
        #self.labels = self.labels[:100]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224), interpolation=2),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            #torchvision.transforms.RandomRotation(180, expand=False),
            torchvision.transforms.RandomGrayscale(0.1),
            torchvision.transforms.ToTensor(),

        ])

    def __len__(self):
        return len(self.picpaths)

    def __getitem__(self, index):
        picpath = self.picpaths[index]
        img = Image.open(picpath).convert('RGB')
       # print(img.size())
        img = self.transforms(img)
        label = self.labels[index]
        target = torch.zeros(self.nclass, dtype=torch.long)
        for intx in label:
            target[intx] = 1
        return img, target




