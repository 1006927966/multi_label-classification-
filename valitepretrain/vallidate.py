import sys
import os
curpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.split(curpath)[0]
print(rootpath)
sys.path.append(rootpath)

import torch

#import matplotlib
import os
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import shutil
import cv2

from src.models.resnext import make_model
# we should change the txtpath get new classes list
def getclasseslist():
    classlist = []
    txtpath = "/data/wujilong/datas/openImg/coarse_train_label.txt"
    with open(txtpath, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        classlist.append(line.split(":")[-1])
    return classlist


# to getmodel we shuold change the config/value_config numclass and pretrained
# resulttxt, modelpath, picdir should change
def main():
    print('ASL Example Inference code on a single image')

    # parsing args
    resulttxt = "/data/wujilong/huangye/292_shinei.txt"
    modelpath = "/data/wujilong/model/ASL/model_292_9.pth"
    picdir = "/data/wujilong/huangye/室内/"
    if os.path.exists(resulttxt):
        os.remove(resulttxt)
    model = make_model("resnext50_32x4d")
    # setup model
    print('creating and loading the model...')
    state = torch.load(modelpath, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    model.cuda()
    classes_list = np.array(getclasseslist())

    print('done\n')

    # doing inference
    names = os.listdir(picdir)
    for name in names:
        pic_path= os.path.join(picdir, name)
        print(pic_path)
        print('loading image and doing inference...')
        try:
            im = Image.open(pic_path).convert("RGB")
            im_resize = im.resize((224, 224))
            np_img = np.array(im_resize, dtype=np.uint8)
            tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
            tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
            output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
            np_output = output.cpu().detach().numpy()
            print(np_output.shape)
            detected_classes = classes_list[np_output > 0.5]
            print(detected_classes)
        except:
            os.remove(pic_path)
            continue
        objects = pic_path
        for object in detected_classes:
            objects += ",{}".format(object)
        with open(resulttxt, "a") as f4:
            f4.write(objects+"\n")
    print('done\n')
    print('showing image on screen...')
    print('done\n')


if __name__ == '__main__':
    main()