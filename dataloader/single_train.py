from torch.utils import data
import torchvision
import os
from PIL import Image
import cv2
import torch

class Traindata(data.Dataset):
    def __init__(self, txtpath):
        self.txtpath = txtpath
        with open(txtpath, "r") as f:
            lines = f.readlines()
        self.picpaths = []
        self.labels = []
        for line in lines:
            line = line.strip()
            factors = line.split(',')
            self.picpaths.append(factors[0])
            self.labels.append(int(factors[1]))
        # self.picpaths = self.picpaths[0:10]
        # self.labels = self.labels[0:10]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224), interpolation=2),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomGrayscale(0.1),
            torchvision.transforms.ToTensor(),

        ])

    def __len__(self):
        return len(self.picpaths)

    def __getitem__(self, index):
        picpath = self.picpaths[index]
        picpath = picpath.replace("/code", "/mnt/wfs")
        img = Image.open(picpath).convert('RGB')
       # print(img.size())
        img = self.transforms(img)
        label = self.labels[index]
        return img, label




