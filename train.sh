#!/bin/bash
#$ -q gpu
#$ -l gpu=1
#$ -M nsapkota@nd.edu
#$ -m abe
#$ -r y
#$ -o crclogs

conda activate pyt

export CUDA_VISIBLE_DEVICES="${SGE_HGR_gpu_card// /,}"
if [ -z ${SGE_HGR_gpu_card+x} ]; then 
        SGE_HGR_gpu_card=-1
fi

python train.py -cnf 'config.yaml'
conda deactivate
