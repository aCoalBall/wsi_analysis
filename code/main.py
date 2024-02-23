from tasks.get_features import *
from tasks.get_similarity import *
from tasks.draw_figures import *
from models.get_models import *
from constants import *
import argparse
import random
import torch


def get_args_parser():
    parser = argparse.ArgumentParser('Experiments', add_help=False)
    parser.add_argument('--task', default='features', type=str)
    parser.add_argument('--model', default='resnet_imgn', type=str)
    parser.add_argument('--dataset', default='camelyon16', type=str)
    return parser

def main(args):
    #fix seeds
    random.seed(42)

    #get features
    if args.task == 'features':
        model = model_dict[args.model]()
        wsi_dir = wsi_dict[args.dataset]
        h5_dir = h5_dict[args.dataset]
        save_features(model=model, model_name=args.model, dataset_name=args.dataset, wsi_dir=wsi_dir, h5_dir=h5_dir)
    
    #get partitioned features (tumor, normal, boundary)
    elif args.task == 'features_partition':
        model = model_dict[args.model]()
        wsi_dir = wsi_dict[args.dataset]
        h5_dir = h5_dict[args.dataset]
        annot_dir = annotation_dict[args.dataset]
        save_features_partitioned(model=model, model_name=args.model, dataset_name=args.dataset, annotation_dir=annot_dir, h5_dir=h5_dir, wsi_dir=wsi_dir)
    
    #get overlapping features
    elif args.task == 'features_overlap':
        model = model_dict[args.model]()
        wsi_dir = wsi_dict[args.dataset]
        h5_dir = h5_dict[args.dataset]
        annot_dir = annotation_dict[args.dataset]
        partition_tissues_whole_dataset_overlapping(model=model, model_name=args.model, dataset_name=args.dataset, annotation_dir=annot_dir, h5_dir=h5_dir, wsi_dir=wsi_dir)     

    #get cosine similarity - distance relations
    elif args.task == 'distance':
        get_distance_similarity_relation(dataset=args.dataset, model=args.model, sampling_points=15)

    #get cosine similarity - distance relations by partition
    elif args.task == 'distance_partition':
        get_distance_similarity_relation_by_tissue(dataset=args.dataset, model=args.model, sampling_points=15)

    #draw all figures
    elif args.task == 'draw':
        draw_figures()
    



if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    args = get_args_parser()
    args = args.parse_args()
    main(args)