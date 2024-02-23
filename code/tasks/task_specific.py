import torch
import torch.nn as nn
import os
from tasks.get_features import load_features
from tasks.dataset import Feature_Dataset
from torch.utils.data import DataLoader
from constants import *


def linear_prob(model_name):
    encoder = model_dict[model_name]
    encoder.to(DEVICE)
    encoder.eval()
    model = nn.Sequential(encoder, nn.Linear(1024, 2))

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    dataset_path = os.path.join('data/camelyon16_partitioned', model_name)
    slides_path = [os.path.join(dataset_path, s) for s in os.listdir(dataset_path)]
    for s in slides_path:
        normal_path = os.path.join(s, 'normal.pth')
        tumor_path = os.path.join(s, 'tumor.pth')
        boundary_path = os.path.join(s, 'boundary.pth')
        normal = load_features(normal_path)
        tumor = load_features(tumor_path) + load_features(boundary_path)
        dataset = Feature_Dataset(normal_features=normal, tumor_features=tumor)
        loader =  DataLoader(dataset, batch_size=1, shuffle=True)
        for i, (sample, label) in enumerate(loader):
            pred = model(sample)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


    correct = 0
    total = 0

    dataset_path = os.path.join('data/camelyon16_test_partitioned', model_name)
    slides_path = [os.path.join(dataset_path, s) for s in os.listdir(dataset_path)]

    for s in slides_path:
        normal_path = os.path.join(s, 'normal.pth')
        tumor_path = os.path.join(s, 'tumor.pth')
        boundary_path = os.path.join(s, 'boundary.pth')
        normal = load_features(normal_path)
        tumor = load_features(tumor_path) + load_features(boundary_path)
        dataset = Feature_Dataset(normal_features=normal, tumor_features=tumor)
        loader =  DataLoader(dataset, batch_size=1, shuffle=True)
        for i, (sample, label) in enumerate(loader):
            pred = model(sample)
            correct += torch.sum(pred == label)
            total += len(dataset)
    acc = correct / total
    print(acc)