import os
import torch
import torch.nn.parallel
import torch.optim
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP
from src.models.model_select import getmodel
from src.loss_functions.losses import AsymmetricLoss
from config.value_config import *
from dataloader.loader_select import get_loader
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"]=GPUS
os.makedirs(SAVEPATH, exist_ok=True)


def validate_multi(val_loader, model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        targets.append(target.cpu().detach())
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    print("mAP score regular {:.2f}".format(mAP_score_regular))
    return mAP_score_regular


# Setup model
print('creating model...')
model = getmodel()
if LOADDING:  # make sure to load pretrained ImageNet model
    state = torch.load(WEIGHT, map_location='cpu')
    model.load_state_dict(state)
    print('done\n')

#  Data loading
trainloader, testloader, valloader = get_loader()


#ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
if torch.cuda.device_count()>1:
    model = DataParallel(model)
model.cuda()
print("model is loadding!!!")
# set optimizer

steps_per_epoch = len(trainloader)
criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)
if FINE:
    parameters = filter(lambda p :p.requires_grad, model.parameters())
else:
    parameters = model.parameters()
optimizer = torch.optim.Adam(params=parameters, lr=LR, weight_decay=0)  # true wd, filter_bias_and_bn
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                                        pct_start=0.2)
highest_mAP = 0
#trainInfoList = []
scaler = GradScaler()
print(len(trainloader))
print(len(testloader))

print("begin train ....")
for epoch in range(EPOCHS):
    model.train()
    for i, (inputData, target) in enumerate(trainloader):
        inputData = inputData.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)  # (batch,3,num_classes)
#        target = target.max(dim=1)[0]
        with autocast():
            output = model(inputData)  # sigmoid will be done in loss !
        loss = criterion(output, target)
        model.zero_grad()
        #loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #optimizer.step()
        scheduler.step()

# store information
        if i%DISFREQ==0:
            print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'.format(epoch, EPOCHS, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], loss.item()))
        if i%STEPSAVEFREQ == 0:
            try:
                torch.save(model.state_dict(), os.path.join(SAVEPATH, 'model-{}-{}.pth'.format(epoch + 1, i)))
            except:
                pass

        if i%STEPDISFREQ == 0:
            model.eval()
            mAP_score = validate_multi(valloader, model)
            print('ecpoch{} step {} current_mAP = {:.2f} '.format(epoch, i, mAP_score))
            model.train()

    if epoch%SAVEFREQ == 0:
        try:
            torch.save(model.state_dict(), os.path.join(SAVEPATH, 'model-{}.pth'.format(epoch + 1)))
        except:
            pass
    model.eval()
    mAP_score = validate_multi(testloader, model)
    if mAP_score > highest_mAP:
        highest_mAP = mAP_score
        try:
            torch.save(model.state_dict(), os.path.join(SAVEPATH, 'model-highest.pth'))
        except:
            pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))

