import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import random
import numpy as np

def process_cosine_similarity_data(oringinal_data):
    overlapping_cs_overall_list = []

    for i in oringinal_data:
        for tissue in i:
            overlapping_cs_overall_list += i[tissue]

    data = [[] for _ in range(11)]
    for i in range(11):
        for j in range(len(overlapping_cs_overall_list)):
            data[i].append(overlapping_cs_overall_list[j][i])
    return data


def to_dataframe(different_models_data, value:str):
    data = []
    for model in different_models_data:
        overlapping_cs = different_models_data[model]
        for overlapping_proportion in range(len(overlapping_cs)):
            if overlapping_proportion % 1 == 0:
                cosine_sim_points = overlapping_cs[overlapping_proportion]
                cosine_sim_points = random.sample(cosine_sim_points, 50000)
                for points in cosine_sim_points:
                    data.append([points, ((10 - overlapping_proportion) * 10), model])
    columns = [value, 'overlapping', 'model']
    data = pd.DataFrame(data, columns = columns)
    #data = data[data['cosine_sim'] > 0.9]
    print('%s loaded'%value)
    return data



def load_pth(value:str):
    if value == 'cosine_sim':
    #cosine similarity, [slides -> tissues -> patch[overlapping 0-1]]
        resnet_cs = process_cosine_similarity_data(torch.load('data/similarity_data/resnet_cs.pth'))
        retccl_cs = process_cosine_similarity_data(torch.load('data/similarity_data/retccl_cs.pth'))
        simclr_cs = process_cosine_similarity_data(torch.load('data/similarity_data/simclr_cs.pth'))
        data = {'resnet' : resnet_cs, 'retccl':retccl_cs, 'simclr':simclr_cs}

    elif value == 'cka':
        resnet_cka = torch.load('data/similarity_data/resnet_cka.pth')
        retccl_cka = torch.load('data/similarity_data/retccl_cka.pth')
        simclr_cka = torch.load('data/similarity_data/simclr_cka.pth')
        data =  {'resnet' : resnet_cka, 'retccl':retccl_cka, 'simclr':simclr_cka}

    elif value == 'eucli':
        resnet_eucli = torch.load('data/similarity_data/resnet_eucli.pth')
        retccl_eucli = torch.load('data/similarity_data/retccl_eucli.pth')
        simclr_eucli = torch.load('data/similarity_data/simclr_eucli.pth')

        data = {'resnet' : resnet_eucli, 'retccl':retccl_eucli, 'simclr':simclr_eucli}

    return data

def draw_distance_figure(distance_data_path, model, tissue, ax):
    sim = reform_distance_data(distance_data_path, model)
    columns = ['similarity', 'distance', 'model']
    sim = pd.DataFrame(sim, columns=columns)
    # Create the box plot
    sns.boxplot(sim, x='distance', y='similarity', showfliers=False, ax=ax)
    ax.set_title(model)
    #plt.boxplot(binned_similarity[:40], positions=np.arange(1, 41), showfliers=False)



def reform_distance_data(distance_data_path, model):
    distance_data = torch.load(distance_data_path)
    #num_bins = 100
    distance_data = random.sample(distance_data, 200000)
    # Separate the data into two lists
    distance, similarity = zip(*distance_data)

    # Bin the x values
    distance = np.array([int(x / 224.0) for x in distance])
    similarity = np.array(list(similarity))
    # Group the y values based on the bins
    binned_similarity = [similarity[distance == i] for i in range(1, max(distance))]
    print('len of binned_similarity :', len(binned_similarity))
    binned_similarity = binned_similarity[:25]
    reformed_binned_similarity = []
    for i in range(len(binned_similarity)):
        binned = binned_similarity[i]
        x = [[b, i, model] for b in binned]
        #print(len(x))
        reformed_binned_similarity += x
    #reformed_binned_similarity = pd.DataFrame(reformed_binned_similarity)
    print(len(reformed_binned_similarity))
    return reformed_binned_similarity

def distance_data_to_dataframe(path='data/similarity_data/distances/camelyon16'):
    resnet_path = os.path.join(path, 'resnet_imgn/distance.pth')
    simclr_path = os.path.join(path, 'resnet_ccl/distance.pth')
    retccl_path = os.path.join(path, 'resnet_simclr/distance.pth')

    resnet_sim = reform_distance_data(resnet_path, 'resnet')
    retccl_sim = reform_distance_data(retccl_path, 'retccl')
    simclr_sim = reform_distance_data(simclr_path, 'simclr')

    sim = resnet_sim + simclr_sim + retccl_sim
    print(len(sim))
    columns = ['similarity', 'distance', 'model']
    sim = pd.DataFrame(sim, columns=columns)
    print(sim)
    return sim

def reform_distance_data_partition(distance_data_path, tissue):
    distance_data = torch.load(distance_data_path)
    #num_bins = 100
    if len(distance_data) > 200000:
        distance_data = random.sample(distance_data, 200000)
    # Separate the data into two lists
    distance, similarity = zip(*distance_data)

    # Bin the x values
    distance = np.array([int(x / 224.0) for x in distance])
    similarity = np.array(list(similarity))
    # Group the y values based on the bins
    binned_similarity = [similarity[distance == i] for i in range(1, max(distance))]
    print('len of binned_similarity :', len(binned_similarity))
    binned_similarity = binned_similarity[:25]
    reformed_binned_similarity = []
    for i in range(len(binned_similarity)):
        binned = binned_similarity[i]
        x = [[b, i, tissue] for b in binned]
        #print(len(x))
        reformed_binned_similarity += x
    #reformed_binned_similarity = pd.DataFrame(reformed_binned_similarity)
    print(len(reformed_binned_similarity))
    return reformed_binned_similarity

def distance_data_to_dataframe_partition(path='../data/similarity_data/distances_partition/camelyon16'):
    resnet_path = os.path.join(path, 'resnet_imgn/distance_tumor.pth')
    simclr_path = os.path.join(path, 'resnet_ccl/distance_tumor.pth')
    retccl_path = os.path.join(path, 'resnet_simclr/distance_tumor.pth')

    resnet_sim_tumor = reform_distance_data_partition(resnet_path, 'tumor')
    retccl_sim_tumor = reform_distance_data_partition(retccl_path, 'tumor')
    simclr_sim_tumor = reform_distance_data_partition(simclr_path, 'tumor')

    resnet_path = os.path.join(path, 'resnet_imgn/distance_normal.pth')
    simclr_path = os.path.join(path, 'resnet_ccl/distance_normal.pth')
    retccl_path = os.path.join(path, 'resnet_simclr/distance_normal.pth')

    resnet_sim_normal = reform_distance_data_partition(resnet_path, 'normal')
    retccl_sim_normal = reform_distance_data_partition(retccl_path, 'normal')
    simclr_sim_normal = reform_distance_data_partition(simclr_path, 'normal')

    resnet_sim = resnet_sim_tumor + resnet_sim_normal
    retccl_sim = retccl_sim_tumor + retccl_sim_normal
    simclr_sim = simclr_sim_tumor + simclr_sim_normal

    columns = ['similarity', 'distance', 'tissue']
    resnet_sim = pd.DataFrame(resnet_sim, columns=columns)
    retccl_sim = pd.DataFrame(retccl_sim, columns=columns)
    simclr_sim = pd.DataFrame(simclr_sim, columns=columns)

    return [resnet_sim, retccl_sim, simclr_sim]

def draw_multiple_distance_figure():
    f, axes = plt.subplots(2, 2, figsize=(10, 8))  # Adjust the grid size (2x2) as needed
    draw_distance_figure(distance_data_path='../data/similarity_data/distances/camelyon16/resnet_imgn/distance.pth', model='resnet_imgn', tissue='all', ax=axes[0,0])
    draw_distance_figure(distance_data_path='../data/similarity_data/distances/camelyon16/resnet_ccl/distance.pth', model='resnet_ccl', tissue='all', ax=axes[0,1])
    draw_distance_figure(distance_data_path='../data/similarity_data/distances/camelyon16/resnet_simclr/distance.pth', model='resnet_simclr', tissue='all', ax=axes[1,0])
    plt.savefig('../figures/distance.png')
    plt.clf()

def draw_multiple_distance_figure_partition():
    resnet_sim,  retccl_sim, simclr_sim = distance_data_to_dataframe_partition()

    f, axes = plt.subplots(2, 2, figsize=(10, 8))  # Adjust the grid size (2x2) as needed

    ax = axes[0, 0]
    sns.boxplot(data=resnet_sim, x='distance', y='similarity', hue='tissue', showfliers=False, ax=ax)
    ax.set_title('ResNet_imgN')

    ax = axes[0, 1]
    sns.boxplot(data=retccl_sim, x='distance', y='similarity', hue='tissue', showfliers=False, ax=ax)
    ax.set_title('ResNet_CCL')

    ax = axes[1, 0]
    sns.boxplot(data=simclr_sim, x='distance', y='similarity', hue='tissue', showfliers=False, ax=ax)
    ax.set_title('ResNet_SimClr')

    #plt.xticks([])
    plt.savefig('../figures/distance_partition.png')
    plt.clf()

    
def draw_avg_imgnet_wsi():
    imgnet_sim = torch.load('data/similarity_data/imgnet_random_cs.pth')
    wsi_sim = torch.load('data/similarity_data/resnet_cs_random.pth')
    combined_data = imgnet_sim + wsi_sim
    labels = ['ImageNet'] * len(imgnet_sim) + ['Camelyon16'] * len(wsi_sim)
    # Create a DataFrame for Seaborn
    df = pd.DataFrame({'Cosine Similarity': combined_data, 'Dataset': labels})
    # Draw a violin plot
    sns.violinplot(x='Dataset', y='Cosine Similarity', data=df)
    plt.yticks([])
    plt.savefig('figures/resnet_cs_img_wsi.png')
    plt.clf()
    

def draw_figures():

    #After using an ImageNet-pretrained ResNet50, the similarity between general images and the similarity between WSI images 
    print('drawing supervised resnet cosine similarity difference', flush=True)
    #draw_avg_imgnet_wsi()

    #Draw distance figures partition
    print('drawing distance figures by partition', flush=True)
    draw_multiple_distance_figure_partition()

    #Draw distance figures 
    print('drawing distance figures', flush=True)
    draw_multiple_distance_figure()

    #Draw Cosine Sim
    '''
    print('drawing overlapping cosine similarity', flush=True)
    data = load_pth('cosine_sim')
    df = to_dataframe(data, 'cosine_sim')
    sns.boxplot(data=df, x="overlapping", y='cosine_sim', hue='model', showfliers=False)
    plt.xlabel('overlapping percent')
    plt.savefig('figures/cs_violin.png')
    plt.clf()
    '''

    