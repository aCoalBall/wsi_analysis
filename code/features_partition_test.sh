#!/usr/local/bin/nosh

#$ -S /usr/local/bin/nosh
#$ -cwd

source ~/venvs/mae/bin/activate

OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model resnet_imgn --dataset camelyon16_test
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model resnet_imgn_moco --dataset camelyon16_test
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model resnet_simclr --dataset camelyon16_test
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model resnet_ccl --dataset camelyon16_test
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model vit_dino --dataset camelyon16_test
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model vit_moco --dataset camelyon16_test