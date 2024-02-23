import torchvision.transforms as transforms
from models.get_models import *
from tasks.dataset import GrayscaleToRGB

TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

IMGNET_TRANSFORM = transforms.Compose([
            GrayscaleToRGB(),    
            transforms.Resize((224,224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

DEVICE = torch.device('cuda')

model_dict = {
    'resnet_imgn' : resnet_baseline,
    'resnet_imgn_moco' : resnet_mocov3,
    'resnet_simclr': resnet_simclr,
    'resnet_ccl' : resnet_retccl,
    'vit_dino' : vit_dino,
    'vit_moco' : vit_mocov3
}

wsi_dict = {
    'camelyon16': '/home/coalball/singularity_files/sandbox/clam/wsi/CAMELYON16/train_set/train_images',
    'camelyon16_test': '/home/coalball/singularity_files/sandbox/clam/wsi/CAMELYON16/test_set/test_images'
}

h5_dict = {
    'camelyon16': '/home/coalball/singularity_files/sandbox/clam/wsi/CAMELYON16/train_set/patches_x20_224/patches',
    'camelyon16_test': '/home/coalball/singularity_files/sandbox/clam/wsi/CAMELYON16/test_set/patches_x20_224/patches'
}

annotation_dict = {
    'camelyon16': '/home/coalball/singularity_files/sandbox/clam/wsi/CAMELYON16/annotations',
    'camelyon16_test': '/home/coalball/singularity_files/sandbox/clam/wsi/CAMELYON16/annotations'
}

feature_dict = {
    'camelyon16': '../data/camelyon16',
    'imgnet' : '../data/imgnet/resnet_features.pth',
    'camelyon16_test': '../data/camelyon16_test'
}

