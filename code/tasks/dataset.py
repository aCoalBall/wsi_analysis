from __future__ import print_function, division
import os
from typing import List
from random import randrange
import random
import torch
import numpy as np
import pandas as pd
import openslide
from PIL import Image
import h5py
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Whole_Slide_Bag_Pair(Dataset):
    def __init__(self, h5, wsi, transform, stride = 0):
        self.h5_path = h5
        self.wsi = openslide.open_slide(wsi)
        self.transform = transform
        self.pairs = []
        self.stride = stride
        with h5py.File(self.h5_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            for i in range(len(dset)):
                coord = f['coords'][i]
                coord_right = coord[0] + int((self.patch_size) * stride), coord[1]
                #coord_above = coord[0], coord[1] + int((self.patch_size) * stride)
                self.pairs.append([coord, coord_right])
        self.length = len(self.pairs)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        coord, coord_right = self.pairs[index]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img_right = self.wsi.read_region(coord_right, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            img_right = self.transform(img_right)
            
        return img, img_right, coord
    
    def close(self):
        self.wsi.close()


class Whole_Slide_Coords(Dataset):
    def __init__(self, coord_list, wsi, transform, patch_size=224, patch_level=1):
        self.coord_list = coord_list
        self.wsi = openslide.open_slide(wsi)
        self.transform = transform
        self.length = len(self.coord_list)
        self.patch_size = patch_size
        self.patch_level = patch_level

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        coord = self.coord_list[index]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, coord
    
    def close(self):
        self.wsi.close()


class Whole_Slide_Coords_Overlapping(Dataset):
    def __init__(self, coord_list, wsi, transform, patch_size=224, patch_level=1):
        self.wsi = openslide.open_slide(wsi)
        self.transform = transform
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.coord_list = []
        for c in coord_list:
            if c[0] - self.patch_size > 0:
                self.coord_list.append(c)
        self.length = len(self.coord_list)
    
    def apply_shift(self, coord, shift):
        direction = random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up':
            coord = coord[0] - shift, coord[1]
        elif direction == 'left':
            coord = coord[0], coord[1] - shift
        elif direction == 'down':
            coord = coord[0] + shift, coord[1]
        else:
            coord = coord[0], coord[1] + shift
        return coord

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        coord = self.coord_list[index]
        overlapping_imgs = []
        for stride in range(11):
            #coord_shift = coord[0] - int((self.patch_size) * stride * 0.1), coord[1]
            coord_shift = self.apply_shift(coord=coord, shift=int((self.patch_size) * stride * 0.1))
            img = self.wsi.read_region(coord_shift, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            img = img.unsqueeze(0)
            overlapping_imgs.append(img)
        overlapping_imgs = torch.cat(overlapping_imgs, dim=0)

        return overlapping_imgs

    def close(self):
        self.wsi.close()


class ImageNet_Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.img_paths = [os.path.join(path, p) for p in os.listdir(path)]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img
    

class GrayscaleToRGB(object):
    def __call__(self, image):
        return image.convert('RGB')
    

class Feature_Dataset(Dataset):
    def __init__(self, normal_features, tumor_features):
        normal_features = [[n[0], 0] for n in normal_features]
        tumor_features = [[t[0], 1] for t in tumor_features]
        self.features = normal_features + tumor_features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index][0], self.features[index][1]