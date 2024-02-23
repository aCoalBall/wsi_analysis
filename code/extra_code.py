from code.get_features import *
from code.get_similarity import *
from code.draw_figures import *
from models.get_models import *
from constants import *


def imagenet_features():
    resnet_imgn = model_dict['resnet_imgn']()
    save_imgnet_features(model=resnet_imgn, model_name='resnet_imgn', dataset_path='data/imgnet/imgs')


def random_cosine_similarity():
    
    get_camelyon16_random_similarity(model_name='resnet_imgn')
    '''
    get_camelyon16_random_similarity(model_name='resnet_imgn_moco')
    get_camelyon16_random_similarity(model_name='resnet_simclr')
    get_camelyon16_random_similarity(model_name='resnet_ccl')
    get_camelyon16_random_similarity(model_name='vit_dino')
    get_camelyon16_random_similarity(model_name='vit_moco')
    '''
    #get_imagenet_random_similarity()

random_cosine_similarity()
