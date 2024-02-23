import torch
from models.get_models import *
from tasks.draw_figures import *
from tasks.reconstruction import reconstruction_test, Decoder
def test_vit_moco_v3():
    model = vit_mocov3()
    x = torch.zeros(2, 3,224,224)
    y = model(x)
    print(y.shape)

def test_imgnet():
    imgnet_sim = torch.load('data/similarity_data/imgnet_random_cs.pth')
    print(imgnet_sim)

def test_reconstruction():
    reconstruction_test(model_name='resnet_imgn', num_epochs=1)

def test_decoder():
    decoder = Decoder()
    x = torch.zeros([1,1024])
    y = decoder(x)
    print(y.shape)

test_reconstruction()
