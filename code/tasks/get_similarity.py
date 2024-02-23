import torch
import torch.nn.functional as F
import random
import os
import numpy as np
from tasks.get_features import load_features
from constants import feature_dict

'''
get the cosine similarity and distances between patches

input: 
dataset: 'camelyon16' by default
model: the name of the feature extractor
sampling_points: the number of patches selected to be compared by all other patches in one wsi
'''
def get_distance_similarity_relation(dataset, model, sampling_points=15):
    #fix the seed for reproducibility
    random.seed(42)
    dataset_path = feature_dict[dataset]
    os.makedirs('../data/similarity_data/distances/%s/%s/'%(dataset, model), exist_ok=True)
    record = []
    path = os.path.join(dataset_path, model)
    slides_feature_path = [os.path.join(path, slide, 'reps.pth') for slide in os.listdir(path)]
    for p in slides_feature_path:
        print(p)
        features = load_features(p)
        print(len(features), flush=True)
        #random sampling patches for each slide
        selected_features = random.sample(features, sampling_points)
        for f in selected_features:
            selected_feature, selected_coord = f
            selected_coord = selected_coord.numpy()
            for patch in features:
                [feature, coord] = patch
                coord = coord.numpy()
                d = np.linalg.norm(coord - selected_coord)
                similarity = F.cosine_similarity(torch.tensor(feature), torch.tensor(selected_feature)).item()
                record.append((d, similarity))
            print(len(record), flush=True)
    torch.save(record, f='../data/similarity_data/distances/%s/%s/distance.pth'%(dataset, model))

'''
get the cosine similarity and distances between patches by tissues (tumor, normal, boundary)

input: 
dataset: 'camelyon16' by default
model: the name of the feature extractor
sampling_points: the number of patches selected to be compared by all other patches in one wsi
'''
def get_distance_similarity_relation_by_tissue(dataset, model, sampling_points=15):
    random.seed(42)
    dataset_path = '../data/camelyon16_partitioned'
    os.makedirs('../data/similarity_data/distances_partition/%s/%s/'%(dataset, model), exist_ok=True)
    record = []
    path = os.path.join(dataset_path, model)
    tumor_feature_path = [os.path.join(path, slide, 'tumor.pth') for slide in os.listdir(path)]
    normal_feature_path = [os.path.join(path, slide, 'normal.pth') for slide in os.listdir(path)]
    for p in tumor_feature_path:
        print(p)
        features = load_features(p)
        print(len(features))
        if len(features) <= sampling_points:
            continue
        #random sampling patches for each slide
        selected_features = random.sample(features, sampling_points)
        for f in selected_features:
            selected_feature, selected_coord = f
            selected_coord = selected_coord.numpy()
            for patch in features:
                [feature, coord] = patch
                coord = coord.numpy()
                d = np.linalg.norm(coord - selected_coord)
                similarity = F.cosine_similarity(torch.tensor(feature), torch.tensor(selected_feature)).item()
                record.append((d, similarity))
    torch.save(record, f='../data/similarity_data/distances_partition/%s/%s/distance_tumor.pth'%(dataset, model))
    for p in normal_feature_path:
        print(p)
        features = load_features(p)
        print(len(features))
        if len(features) <= sampling_points:
            continue
        #random sampling patches for each slide
        selected_features = random.sample(features, sampling_points)
        for f in selected_features:
            selected_feature, selected_coord = f
            selected_coord = selected_coord.numpy()
            for patch in features:
                [feature, coord] = patch
                coord = coord.numpy()
                d = np.linalg.norm(coord - selected_coord)
                similarity = F.cosine_similarity(torch.tensor(feature), torch.tensor(selected_feature)).item()
                record.append((d, similarity))
    torch.save(record, f='../data/similarity_data/distances_partition/%s/%s/distance_normal.pth'%(dataset, model))





'''
get the cosine similarity for random chosen images from the validation set of ImageNet
'''
def get_imagenet_random_similarity():
    features = load_features(path=feature_dict['imgnet'])
    print(len(features))
    if len(features) % 2 == 1:
        features = features[:-1]
    random.shuffle(features)
    pairs = [(features[i], features[i + 1]) for i in range(0, len(features), 2)]
    similarities = []
    for x, y in pairs:
        cs = F.cosine_similarity(x, y)
        cs = cs.item()
        similarities.append(cs)
    torch.save(similarities, '../data/similarity_data/imgnet_random_cs.pth')

'''
get the cosine similarity for random chosen patches from Camelyon16
'''
def get_camelyon16_random_similarity(model_name):
    path = os.path.join(feature_dict['camelyon16'], model_name)
    slides_path = [os.path.join(path, p) for p in os.listdir(path)]
    similarities = []
    for slide_path in slides_path:
        random_cs_one_image = get_random_cosine_sim_single_image(slide_path)
        similarities += random_cs_one_image
    similarities = random.sample(similarities, 200000)
    torch.save(similarities, '../data/similarity_data/resnet_cs_random.pth')

'''
get the cosine similarity for random chosen patches from single wsi
'''
def get_random_cosine_sim_single_image(feature_path):
    feature_path = os.path.join(feature_path, 'reps.pth')
    features = torch.load(feature_path)
    if len(features) % 2 == 1:
        features = features[:-1]
    random.shuffle(features)
    pairs = [(features[i], features[i + 1]) for i in range(0, len(features), 2)]
    all_cs = []
    for x, y in pairs:
        x, y = x[0], y[0]
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        cs = F.cosine_similarity(x, y)
        cs = cs.item()
        all_cs.append(cs)
    return all_cs


