#!/usr/local/bin/nosh

#$ -S /usr/local/bin/nosh
#$ -cwd

source ~/venvs/mae/bin/activate

OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model resnet_imgn
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model resnet_imgn_moco
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model resnet_simclr
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model resnet_ccl
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model vit_dino
OMP_NUM_THREADS=1 python3 main.py  --task features_partition --model vit_moco