import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
#import matplotlib
import os
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import shutil
import cv2

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')
parser.add_argument('--model_path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
parser.add_argument('--pic_dir', type=str, default='./pics/')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--dataset_type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.95)
parser.add_argument('--txt_path', type=str, default="./result.txt")
parser.add_argument('--subdir', type=str, default="./pics")
#parser.add_argument('--result_txt', type=str, default="./result.txt")

def main():
    print('ASL Example Inference code on a single image')

    # parsing args
    args = parse_args(parser)
    resulttxt = args.txt_path
    if os.path.exists(resulttxt):
        os.remove(resulttxt)
    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))

    print('done\n')

    # doing inference
    os.makedirs(args.subdir, exist_ok=True)
    savedir = args.subdir
    dir = args.pic_dir
    if "jpg" not in os.listdir(dir)[0] and "jpeg" not in os.listdir(dir)[0] and "png" not in os.listdir(dir)[0]:
        subdirs = []
        for subdir in os.listdir(dir):
            #if subdir not in ["单人","拍照","多人","其他","室内","无人","制图"]:
            subdirs.append(os.path.join(dir, subdir))
        #subdirs = [os.path.join(dir, subdir) for subdir in os.listdir(dir)]
    else:
        subdirs = [dir]
    for subdir in subdirs:
        taglist = []
        _, subdirname = os.path.split(subdir)
        names = os.listdir(subdir)
        for name in names:
            pic_path= os.path.join(subdir, name)
            print(pic_path)
            objects = pic_path
            os.makedirs(os.path.join(savedir, subdirname), exist_ok=True)
            savepath = os.path.join(savedir, subdirname, name)
            print('loading image and doing inference...')
            try:
                im = Image.open(pic_path).convert("RGB")
                im_resize = im.resize((args.input_size, args.input_size))
                print(args.input_size)
                np_img = np.array(im_resize, dtype=np.uint8)
                tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
                tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
                output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
                np_output = output.cpu().detach().numpy()
                detected_classes = classes_list[np_output > 0.95]
            except:
                os.remove(pic_path)
                continue
            print(len(classes_list))
            for object in detected_classes:
                print(object)
                if object not in taglist:
                    taglist.append(object)
                objects += ",{}".format(object)
            with open(resulttxt, "a") as f4:
                f4.write(objects+"\n")
            # img = cv2.imread(pic_path)
            # cv2.putText(img, objects, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # cv2.imwrite(savepath, img)
        # savetxt = os.path.join(savedir, subdirname+".txt")
        # for obj in taglist:
        #     with open(savetxt, "a") as f:
        #         f.write("{}\n".format(obj))
    print('done\n')

    print('showing image on screen...')
    print('done\n')


if __name__ == '__main__':
    main()

