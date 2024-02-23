import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#from similarity.tumor_vs_normal import load_type_representations

'''
def knn(model:str):
    [tumor_reps, boundary_reps, normal_reps] = load_type_representations(model)

    random.seed(42)
    random.shuffle(tumor_reps)
    random.shuffle(boundary_reps)
    random.shuffle(normal_reps)


    train_tumor_reps, test_tumor_reps = tumor_reps[10:], tumor_reps[:10] 
    train_boundary_reps, test_boundary_reps = boundary_reps[10:], boundary_reps[:10]
    train_normal_reps, test_normal_reps = normal_reps[10:], normal_reps[:10] 

    tumor_labels = np.full((len(tumor_reps),), 2)
    train_tumor_labels, test_tumor_labels = tumor_labels[10:], tumor_labels[:10]

    boundary_labels = np.full((len(boundary_reps),), 1)
    train_boundary_labels, test_boundary_labels = boundary_labels[10:], boundary_labels[:10]

    normal_labels = np.full((len(normal_reps),), 0)
    train_normal_labels, test_normal_labels = normal_labels[10:], normal_labels[:10]

    train_labels = np.concatenate([train_tumor_labels, train_boundary_labels, train_normal_labels])
    train_data = np.concatenate([train_tumor_reps, train_boundary_reps, train_normal_reps])[:,0,:]
    

    print(train_data.shape)
    print(train_labels.shape)

    print(np.array(test_tumor_reps)[:,0,:].shape)
    knn_classifier = KNeighborsClassifier(n_neighbors=10)
    knn_classifier.fit(train_data, train_labels)

    tumor_pred = knn_classifier.predict(np.array(test_tumor_reps)[:,0,:])
    tumor_accuracy = accuracy_score(test_tumor_labels, tumor_pred)
    print(tumor_accuracy)


    boundary_pred = knn_classifier.predict(np.array(test_boundary_reps)[:,0,:])
    boundary_accuracy = accuracy_score(test_boundary_labels, boundary_pred)
    print(boundary_accuracy)

    normal_pred = knn_classifier.predict(np.array(test_normal_reps)[:,0,:])
    normal_accuracy = accuracy_score(test_normal_labels, normal_pred)
    print(normal_accuracy)
'''



def knn_all(path='saved_representations/camelyon16', model='simclr', neighbors=5):
    path = os.path.join(path, model)
    slides = os.listdir(path)
    print(slides)
    print(model, flush=True)
    sum_tumor_accuracy = 0
    for slide in slides:

        tumor_path = os.path.join(path, '%s/tumor.pth'%slide)
        normal_path = os.path.join(path, '%s/normal.pth'%slide)

        boundary_path = os.path.join(path, '%s/boundary.pth'%slide)

        tumor_reps = torch.load(tumor_path)
        tumor_reps = [i[0].squeeze() for i in tumor_reps]

        boundary_reps =torch.load(boundary_path)
        boundary_reps = [i[0].squeeze() for i in boundary_reps]

        normal_reps = torch.load(normal_path)
        normal_reps = [i[0].squeeze() for i in normal_reps]
        
        print(len(normal_reps), len(tumor_reps), flush=True)
        print(len(normal_reps) + len(tumor_reps) + len(boundary_reps))
        
        random.seed(42)
        random.shuffle(tumor_reps)
        random.shuffle(normal_reps)

        train_tumor_reps, test_tumor_reps = tumor_reps[20:], tumor_reps[:20] 

        tumor_labels = [1,] * len(tumor_reps)
        train_tumor_labels, test_tumor_labels = tumor_labels[20:], tumor_labels[:20]


        normal_labels = [0,] * len(normal_reps)
        
        train_labels = train_tumor_labels + normal_labels
        train_data = train_tumor_reps + normal_reps

        knn_classifier = KNeighborsClassifier(n_neighbors=neighbors)
        knn_classifier.fit(train_data, train_labels)

        tumor_pred = knn_classifier.predict(test_tumor_reps)
        tumor_accuracy = accuracy_score(test_tumor_labels, tumor_pred)
        print(tumor_accuracy, flush=True)
        sum_tumor_accuracy += tumor_accuracy

    
    avg_tumor = sum_tumor_accuracy / len(slides)

    print('Tumor Accuarcy ', avg_tumor, flush=True)


    file_path = 'logs/knn_%s_logs.txt'%model
    with open(file_path, 'w') as fp:
        fp.write(str([avg_tumor, ]))



