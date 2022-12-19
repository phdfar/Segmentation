import torch
import torch.nn as nn
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import scipy.sparse
import torch
from skimage.morphology import binary_dilation, binary_erosion
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from accelerate import Accelerator


def get_transform(name: str):
    if any(x in name for x in ('dino', 'mocov3', 'convnext', )):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise NotImplementedError()
    return transform

def get_model(name: str):
    if 'dino' in name:
        model = torch.hub.load('facebookresearch/dino:main', name)
        model.fc = torch.nn.Identity()
        val_transform = get_transform(name)
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
    else:
        raise ValueError(f'Cannot get model: {name}')
    model = model.eval()
    return model, val_transform, patch_size, num_heads


model_name = 'dino_vits16'

which_block=-1
patch_size = 16
output_dict = {}
accelerator = Accelerator(fp16=True, cpu=False)
model, val_transform, patch_size, num_heads = get_model(model_name)

if 'dino' in model_name or 'mocov3' in model_name:
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][which_block]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
else:
    raise ValueError(model_name)
    
    
