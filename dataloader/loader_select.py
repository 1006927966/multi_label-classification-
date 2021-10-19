from torch.utils import data
from config.value_config import *
import os
from src.helper_functions.helper_functions import CocoDetection
import torchvision.transforms as transforms
import torch

key = LOADERNAME
trainpath = os.path.join(DATAPATH, TRAINNAME)
testpath = os.path.join(DATAPATH, TESTNAME)
valpath = os.path.join(DATAPATH, VALNAME)
num = CLSNUM

def get_loader():
    if key == "comslow":
        from dataloader.traindata import Traindata
        from dataloader.valdata import Testdata
        trainset = Traindata(trainpath, num)
        #testset = Traindata(trainpath, num)
        testset = Testdata(testpath, num)
        valset = Testdata(valpath, num)
        trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=True, pin_memory=True)
        testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False, pin_memory=True)
        valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False, pin_memory=True)
        return trainloader, testloader, valloader
    if key == "coco":
        vpath = "/code/wujilong/coco/val2014_224x224"
        tpath = "/code/wujilong/coco/train2014_224x224"
        valfile = "/code/wujilong/coco/annotations/instances_val2014.json"
        tfile = "/code/wujilong/coco/annotations/instances_val2014.json"
        testset = CocoDetection(vpath, valfile, transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
        valset =  CocoDetection(vpath, valfile, transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
        trainset = CocoDetection(tpath, tfile, transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
            # torchvision.transforms.RandomRotation(180, expand=False),
                                    transforms.RandomGrayscale(0.1),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
        trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=True,
                                      pin_memory=True)
        testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False,
                                     pin_memory=False)
        valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False, pin_memory=False)
        return trainloader, testloader, valloader
    if key=="single":
        from dataloader.single_train import Traindata
        from dataloader.single_val import Testdata
        trainset = Traindata(trainpath)
        testset = Testdata(testpath)
        valset = Testdata(valpath)
        trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=True,
                                      pin_memory=True)
        testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False,
                                     pin_memory=True)
        valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False, pin_memory=True)
        return trainloader, testloader, valloader
    if key=="multi_machine":
        from dataloader.traindata import Traindata
        from dataloader.valdata import Testdata
        trainset = Traindata(trainpath, num)
        testset = Testdata(testpath, num)
        valset = Testdata(valpath, num)
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False,
                                      pin_memory=True,sampler=trainsampler)
        testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False,
                                     pin_memory=True)
        valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False, pin_memory=True)
        return trainsampler, trainloader, valloader
    else:
        print('[*]! the key error of dataloader!!!!!')


if __name__ == "__main__":
    _, testloader,_ = get_loader()
    for i , data in enumerate(testloader):
        print(data[1])
