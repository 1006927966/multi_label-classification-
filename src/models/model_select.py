from config.value_config import MODELNAME, PREMODEL
# import argparse
# from src.models import create_model
# from src.helper_functions.helper_functions import parse_args






key = MODELNAME
#key = "resnext50_32x4d"

def getmodel():
    if key == "resnext50_32x4d":
        from src.models.resnext import make_model
        model = make_model(key)
        return model
    if key == "finetun_tres":
        from src.models.finetun_tres import Tres
        if len(PREMODEL) == 0:
            print("Please add the premodel!!!!!!!!")
        model = Tres()
        for param in model.parameters():
            param.requires_grad = False
        model.output.requires_grad_(True)
        return model
    if key == "tres":
        from src.models.finetun_tres import Tres
        if len(PREMODEL) == 0:
            print("Please add the premodel!!!!!!!!")
        model = Tres()
        return model
    else:
        print("[*]!  model key is error!!!!!")


if __name__ == "__main__":
    model = getmodel()
    for k , v in model.named_parameters():
         print(v.requires_grad)

