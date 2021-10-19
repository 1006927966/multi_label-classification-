import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from config.value_config import *
import random
from src.helper_functions.helper_functions import mAP
from src.models.model_select import getmodel
from src.loss_functions.losses import AsymmetricLoss
from dataloader.loader_select import get_loader
from torch.optim import lr_scheduler
import os

def main():
    if SEED is not None:
        random.seed(SEED)
        torch.manual_seed(SEED)
        main_worker()
    # mp.spawn(main_worker, nprocs=torch.cuda.device_count())


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_multi(val_loader, model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        # compute output
        with torch.no_grad():
            output_regular = Sig(model(input.cuda())).cpu()
        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        targets.append(target.cpu().detach())
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    print("mAP score regular {:.2f}".format(mAP_score_regular))
    return mAP_score_regular

def main_worker():
    highest_MAP = 0
    port = os.environ["MASTER_PORT"]
    addr = os.environ["MASTER_ADDR"]
    word_size = os.environ["WORLD_SIZE"]
    rank = os.environ["RANK"]
    dist.init_process_group(backend="nccl", init_method="tcp://"+addr+":"+port, world_size=word_size, rank=rank)
    model = getmodel()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=0)
    trainsampler, trainloader, valloader = get_loader()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=20, eta_min=1e-8, last_epoch=-1)
    for epoch in range(EPOCHS):
        trainsampler.set_epoch(epoch)
        train(trainloader, model, criterion, optimizer, epoch)
        MAP = validate_multi(valloader, model)
        if MAP > highest_MAP:
            highest_MAP = MAP
            torch.save(model.state_dict(), os.path.join(SAVEPATH, "hightst_{:.2f}.pth".format(MAP)))
        torch.save(model.state_dict(), os.path.join(SAVEPATH, "epoch_{}_{:.2f}".format(epoch, MAP)))
        print("[*]! epoch:{} MAP:{:.2f}".format(epoch, MAP))
        scheduler.step()


if __name__ == "__main__":
    main()