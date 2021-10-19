# for data

DATAPATH = "/data/wujilong/datas"
TRAINNAME = "train_ajuke.txt"
TESTNAME = "val_ajuke.txt"
VALNAME = "val_ajuke.txt"
LOADERNAME = "single"
CLSNUM = 8
BATCH_SIZE = 64
THREADS = 16

# for model

MODELNAME = "tres"
#MODELNAME = "resnext50_32x4d"
# for train

LR=1e-4
LOADDING= False
WEIGHT=""
SAVEPATH="/data/wujilong/tres_nofine"
SAVEFREQ = 1
THRESH = 0.8
STEPSAVEFREQ = 1
STEPDISFREQ = 1

EPOCHS = 100
DISFREQ = 1
GPUS='1'
# for fintune
PREMODEL = "/data/wujilong/model/ASL/Open_ImagesV6_TRresNet_L_448.pth"
#PREMODEL = "/home/wujilong/.cache/torch/checkpoints/resnext50_32x4d-7cdf4587.pth"
FINE = False
MAPLIST=["写字楼-公共区", "写字楼-办公区", "卧室", "卫生间", "厨房", "客厅", "室外图", "阳台"]

# for multi machine learning
WORD_SIZE=2
RANK=0
DIST_URL = "tcp://localhost:10001"
BACKEND="nccl"
SEED=108
