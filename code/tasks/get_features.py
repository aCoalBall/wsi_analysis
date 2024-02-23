import os
import torch
import h5py
from tasks.dataset import *
from constants import TRANSFORM
from shapely import Polygon
from lxml import etree
from torch.utils.data import Dataset, DataLoader
from constants import *
import pandas as pd

def get_one_slide_features(model, wsi_path, h5_path, transform=TRANSFORM):
    h5 = h5py.File(h5_path, 'r')
    coords = h5['coords'][:]
    patch_level = h5['coords'].attrs['patch_level']
    patch_size = h5['coords'].attrs['patch_size']
    dataset = Whole_Slide_Coords(coords, wsi=wsi_path, transform=transform, patch_size=patch_size, patch_level=patch_level)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features = []
    for _, sample in enumerate(loader):
        img, coord = sample
        img = img.to(DEVICE)
        r_img = model(img)
        r_img = r_img.detach().to('cpu').numpy()
        features.append((r_img, coord[0]))
    return features

def save_features(model, model_name, dataset_name, wsi_dir:str, h5_dir:str):
    h5s = os.listdir(h5_dir)
    model = model.to(DEVICE)
    model.eval()
    slide_names = []
    for h5 in h5s:
        slide_name = h5[:-3]
        slide_names.append(slide_name)
    os.makedirs('data/%s/%s'%(dataset_name, model_name), exist_ok=True)
    for s in slide_names:
        print(s)
        h5_path = os.path.join(h5_dir, s + '.h5')
        wsi_path = os.path.join(wsi_dir, s + '.tif')
        features = get_one_slide_features(model, wsi_path, h5_path)
        os.makedirs('../data/%s/%s/%s'%(dataset_name, model_name, s), exist_ok=True)
        torch.save(features, f='../data/%s/%s/%s/reps.pth'%(dataset_name, model_name, s))


def partition_tissues_whole_dataset_overlapping(model, model_name, dataset_name, annotation_dir, h5_dir, wsi_dir):
    model = model.to(DEVICE)
    model.eval()
    h5s = os.listdir(h5_dir)
    tumor_slide_names = []
    normal_slide_names = []
    for h5 in h5s:
        slide_name = h5[:-3]
        if slide_name[:5] == 'tumor':
            tumor_slide_names.append(slide_name)
        elif slide_name[:6] == 'normal':
            normal_slide_names.append(slide_name)
    os.makedirs('../data/overlapping/%s/%s'%(dataset_name, model_name), exist_ok=True)
    print(normal_slide_names, flush=True)
    print(tumor_slide_names, flush=True)
    for s in normal_slide_names:
        h5_path = os.path.join(h5_dir, s + '.h5')
        h5 = h5py.File(h5_path, 'r')
        os.makedirs('../data/overlapping/%s/%s/%s'%(dataset_name, model_name, s), exist_ok=True)
        representations = []
        coords = list(h5['coords'][:])
        dataset = Whole_Slide_Coords_Overlapping(coords, wsi=os.path.join(wsi_dir, s + '.tif'), transform=TRANSFORM, patch_size=224, patch_level=1)
        for imgs in dataset:
            if imgs != None:
                imgs = imgs.to(DEVICE)
                r_imgs = model(imgs)
                r_imgs = r_imgs.detach().to('cpu').numpy()
                representations.append(r_imgs)
        torch.save(representations, f='../data/overlapping/%s/%s/%s/normal.pth'%(dataset_name, model_name, s))
        print('%s done'%s, flush=True)
    for s in tumor_slide_names:
        try:
            xml_path = os.path.join(annotation_dir, s + '.xml')
            h5_path = os.path.join(h5_dir, s + '.h5')
            tumor_coords, boundary_coords, normal_coords = partition_tissues(xml_path, h5_path)
            dictionary = {'tumor': tumor_coords, 'boundary':boundary_coords, 'normal':normal_coords}
            for tissue in ['tumor', 'boundary', 'normal']:
                os.makedirs('../data/overlapping/%s/%s/%s'%(dataset_name, model_name, s), exist_ok=True)
                representations = []
                coords = dictionary[tissue]
                dataset = Whole_Slide_Coords_Overlapping(coords, wsi=os.path.join(wsi_dir, s + '.tif'), transform=TRANSFORM, patch_size=224, patch_level=1)
                for imgs in dataset:
                    if imgs != None:
                        imgs = imgs.to(DEVICE)
                        r_imgs = model(imgs)
                        r_imgs = r_imgs.detach().to('cpu').numpy()
                        representations.append(r_imgs)
                torch.save(representations, f='../data/overlapping/%s/%s/%s/%s.pth'%(dataset_name, model_name, s, tissue))
            print('%s done'%s, flush=True)
        except:
            continue


def save_imgnet_features(model, model_name, dataset_path):
    model = model.to(DEVICE)
    model.eval()
    dataset = ImageNet_Dataset(dataset_path, transform=IMGNET_TRANSFORM)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features = []
    for i, img in enumerate(loader):
        img = img.to(DEVICE)
        r_img = model(img)
        r_img = r_img.detach().to('cpu')
        features.append(r_img)
        if i % 10 == 0:
            print(i)
            print(r_img.shape)
    torch.save(features, f='../data/imgnet/resnet_features.pth')



def partition_tissues(xml_path, h5_path, patch_size=224):
    # Load Annotations
    tree = etree.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for annotation_elem in root.findall('.//Annotation'):
        coordinates = []
        for coord_elem in annotation_elem.findall('.//Coordinate'):
            x = float(coord_elem.get('X')) 
            y = float(coord_elem.get('Y'))
            coordinates.append((x, y))
        annotations.append(coordinates)
    #Get polygons (range of tumor tissues)
    polygons = []
    for a in annotations:
        polygon = Polygon(a)
        polygons.append(polygon)
    #Get patches
    h5 = h5py.File(h5_path, 'r')
    coords = h5['coords'][:]
    #Check which patches are inside the tumor tissue, which are on the boudary, which are not
    tumor_coords = []
    boundary_coords = []
    normal_coords = []
    for c in coords:
        c_poly = Polygon([c, [c[0], c[1] + patch_size], [c[0] + patch_size, c[1]], [c[0] + patch_size, c[1] + patch_size]])
        normal = True
        for p in polygons:
            if p.contains(c_poly):
                tumor_coords.append(c)
                normal = False
                break
            elif p.intersects(c_poly):
                boundary_coords.append(c)
                normal = False
                break
        if normal:
            normal_coords.append(c)
    return tumor_coords, boundary_coords, normal_coords


def get_one_slide_features_partitioned(model,
                        model_name,
                        dataset_name,
                        slide_name,
                        tumor_coords, 
                        boundary_coords, 
                        normal_coords, 
                        wsi_path, 
                        patch_size=224,
                        patch_level=1):
    dictionary = {'tumor': tumor_coords, 'boundary':boundary_coords, 'normal':normal_coords}
    for tissue in ['tumor', 'boundary', 'normal']:
        print(tissue)
        os.makedirs('../data/%s_partitioned/%s/%s'%(dataset_name, model_name, slide_name), exist_ok=True)
        representations = []
        coords = dictionary[tissue]
        dataset = Whole_Slide_Coords(coords, wsi=wsi_path, transform=TRANSFORM, patch_size=patch_size, patch_level=patch_level)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for _, sample in enumerate(dataloader):
            img, coord = sample
            img = img.to(DEVICE)
            r_img = model(img)
            r_img = r_img.detach().to('cpu').numpy()
            representations.append((r_img, coord[0]))
        torch.save(representations, f='../data/%s_partitioned/%s/%s/%s.pth'%(dataset_name, model_name, slide_name, tissue))

def save_features_partitioned(model, model_name, dataset_name, annotation_dir, h5_dir, wsi_dir):
    model = model.to(DEVICE)
    model.eval()
    h5s = os.listdir(h5_dir)
    slide_names = []
    for h5 in h5s:
        slide_name = h5[:-3]
        if slide_name[:5] == 'tumor':
            slide_names.append(slide_name)
    os.makedirs('data/%s_partitioned/%s'%(dataset_name, model_name), exist_ok=True)
    for s in slide_names:
        xml_path = os.path.join(annotation_dir, s + '.xml')
        h5_path = os.path.join(h5_dir, s + '.h5')
        try:
            tumor_coords, boundary_coords, normal_coords = partition_tissues(xml_path, h5_path)
            #if len(tumor_coords) >= 40:
            get_one_slide_features_partitioned(model=model, model_name=model_name,
                                                    dataset_name=dataset_name,
                                                    slide_name=s, tumor_coords=tumor_coords,
                                                    boundary_coords=boundary_coords, normal_coords=normal_coords,
                                                    wsi_path=os.path.join(wsi_dir, s + '.tif'), 
                                                    )
        except:
            continue


def load_features(path):
    features = torch.load(path)
    return features

