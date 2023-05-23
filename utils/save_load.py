import copy
import torch

def save(model, head, save_path, epoch, args):
    head_saved = copy.deepcopy(head.module.state_dict()) if head is not None else None
    sd = {
        'model' : copy.deepcopy(model.module.state_dict()),
        "epoch" : epoch,
        'clf'  : head_saved,
        'args'  : args
    }

    torch.save(sd, save_path)
    return sd

def load():
    pass