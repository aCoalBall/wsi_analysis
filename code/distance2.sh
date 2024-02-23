#!/usr/local/bin/nosh

#$ -S /usr/local/bin/nosh
#$ -cwd

source ~/venvs/mae/bin/activate

OMP_NUM_THREADS=1 python3 main.py  --task distance --model vit_moco
OMP_NUM_THREADS=1 python3 main.py  --task distance --model vit_dino
OMP_NUM_THREADS=1 python3 main.py  --task distance --model resnet_imgn_moco
