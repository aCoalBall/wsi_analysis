import torchvision
import torch
import torch.nn as nn
from functools import partial

from models.byol_pytorch.byol_pytorch_get_feature import BYOL

import models.moco.builder_infence
import models.moco.loader
import models.moco.optimizer
import models.vits
from models.hipt_model_utils import get_vit256
from models.resnet_custom import ResNet_Baseline, Bottleneck_Baseline, load_pretrained_weights, load_pretrained_weights_from_local
from models.RetCCL.ResNet import resnet50

def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


def resnet_baseline(pretrained=True):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model

def resnet_mocov3():
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    model = load_pretrained_weights_from_local(model, '../checkpoints/moco.pth')
    return model

def resnet_dino():
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    model = load_pretrained_weights_from_local(model, '../checkpoints/dino.pth')
    return model

def resnet_retccl():
    model = resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
    pretext_model = torch.load('../checkpoints/retccl.pth')
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    return model


def resnet_simclr():  
    model = torchvision.models.__dict__['resnet18']()
    state = torch.load('../checkpoints/simclr.ckpt')
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    model = load_model_weights(model, state_dict)
    model.fc = torch.nn.Sequential()
    return model


def vit_mocov3():
    model = models.moco.builder_infence.MoCo_ViT(
        partial(models.vits.__dict__['vit_small'], stop_grad_conv1=False))
    #model = nn.DataParallel(model).cuda()
    pretext_model = torch.load('../checkpoints/vit_small.pth.tar')['state_dict']
    for key in list(pretext_model.keys()):
        pretext_model[key[7:]] = pretext_model[key]
        del pretext_model[key]
    model.load_state_dict(pretext_model, strict=False)
    return model


def vit_dino():
    model = get_vit256(pretrained_weights='../checkpoints/vit256_small_dino.pth')
    return model