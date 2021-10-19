import os
from torch.nn import DataParallel

from dataloader.loader_select import get_loader
from config.value_config import *
from src.models.model_select import  getmodel
import torch
import time
import numpy as np

from torch.cuda.amp import GradScaler, autocast


os.environ['CUDA_VISIBLE_DEVICES'] = GPUS
os.makedirs(SAVEPATH, exist_ok=True)

def eval_fuse(testloader, net, gpus):
    fuse_matrix = np.zeros((CLSNUM, CLSNUM))
    net.eval()
    total_num = 0
    acc = 0
    for i, data in enumerate(testloader):
        with torch.no_grad():
            if gpus>0:
                img, label = data[0].cuda(), data[1].cuda()
            else:
                img, label = data[0], data[1]
            total_num += img.size(0)
            #with autocast():
            pre = net(img)
            _, prelabel = torch.max(pre, 1)
            for i in range(CLSNUM):
                for j in range(CLSNUM):
                    fuse_matrix[i][j] += torch.sum((label.data==i)&(prelabel.data==j)).item()
            acc += torch.sum(prelabel.data == label.data)
    test_acc = float(acc)/total_num
    recalldic = {}
    precisiondic = {}
    for i in range(CLSNUM):
        t = fuse_matrix[i][i]
        prenum = np.sum(fuse_matrix[:, i])
        num = np.sum(fuse_matrix[i,:])
        recalldic[i] = t/num
        precisiondic[i] = t/prenum
    print(fuse_matrix)
    return test_acc, recalldic, precisiondic


def displaymetric(recalldic, precisiondic):
    for key in recalldic.keys():
        name = MAPLIST[key]
        print('[*]! {} recall is : {}'.format(name, recalldic[key]))
    for key in precisiondic.keys():
        name = MAPLIST[key]
        print('[*]! {} precision is : {}'.format(name, precisiondic[key]))


trainloader, testloader, valloader = get_loader()
beg = time.time()
net = getmodel()
if LOADDING:
    net.load_state_dict(torch.load(WEIGHT, map_location='cpu'))
net.cuda()
if torch.cuda.device_count()>1:
    net = DataParallel(net)
end = time.time()
print('[*]! model load time is{}'.format(end-beg))
iters = len(trainloader)
if not FINE:
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    for param in net.parameters():
        print(param.requires_grad)
    print("User all param for optimizer")
else:
    optimizer = torch.optim.SGD(filter(lambda p :p.requires_grad, net.parameters()), lr=LR, momentum=0.9, weight_decay=1e-4)
    print("User only param for optimizer")
    for param in net.parameters():
        print(param.requires_grad)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min= 1e-8, last_epoch=-1)

print('[*] train start !!!!!!!!!!!')
best_acc = 0
best_epoch = 0
#scaler = GradScaler()
for epoch in range(EPOCHS):
    net.train()
    train_loss = 0
    total = 0
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
        batch_size = img.size(0)
        optimizer.zero_grad()
        pre = net(img)
        loss = torch.nn.CrossEntropyLoss()(pre, label)
        train_loss += loss * batch_size
        total += batch_size
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('[*] epoch:{} - lr:{} - train loss: {:.3f}'.format(epoch, scheduler.get_last_lr()[0], train_loss/total))
    acc, recalldic, precisiondic = eval_fuse(testloader, net, torch.cuda.device_count())
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        os.makedirs(SAVEPATH, exist_ok=True)
        if torch.cuda.device_count() > 1:
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()
        mdname = os.path.join(SAVEPATH, 'model_best.pth')
        if os.path.exists(mdname):
            os.remove(mdname)
        torch.save(net_state_dict, os.path.join(SAVEPATH, 'model_best.pth'))
        print('[*] change the best model')
    print('[*] epoch:{} - test acc: {:.3f} - best acc: {}_{}'.format(epoch, acc, best_epoch, best_acc))
    displaymetric(recalldic, precisiondic)
print('[*] training finished')


