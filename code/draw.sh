#!/usr/local/bin/nosh

#$ -S /usr/local/bin/nosh
#$ -cwd

source ~/venvs/mae/bin/activate

OMP_NUM_THREADS=1 python3 main.py  --task draw