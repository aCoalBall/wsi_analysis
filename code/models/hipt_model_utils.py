### Dependencies
# Base Dependencies
import argparse
import colorsys
from io import BytesIO
import os
import random
import requests
import sys

# LinAlg / Stats / Plotting Dependencies
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
from torchvision import transforms
from einops import rearrange, repeat
torch.multiprocessing.set_sharing_strategy('file_system')

# Local Dependencies
from . import vision_transformer as vits


def get_vit256(pretrained_weights, arch='vit_small', device=torch.device('cuda:0')):
    r"""
    Builds ViT-256 Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    device = torch.device("cpu")
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    for p in model256.parameters():
        p.requires_grad = False
    model256.eval()
    model256.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model256