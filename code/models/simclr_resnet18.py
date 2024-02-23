import torchvision
import torch
import torch.nn as nn



def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

def get_simclr_resnet18():  

    model = torchvision.models.__dict__['resnet18']()
    model.fc = nn.Linear(512, 1024)

    state = torch.load('other_models/checkpoints/simclr.ckpt')
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    model = load_model_weights(model, state_dict)
    model.fc = nn.Linear(512, 1024)
    return model

