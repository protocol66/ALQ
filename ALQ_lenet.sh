#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --mincpus=1
#SBATCH --partition=gpu-warp
#SBATCH --output=./logs/slurm/slurm-%j-lenet.out
. /opt/ohpc/pub/spack/0.18.1/share/spack/setup-env.sh
spack load miniconda3 cudnn@8.2.4 cuda@11.7
source activate Tensorflow
python lenet5.py --PRETRAIN --ALQ --POSTTRAIN --data ~/Datasets/mnist --model_ori models/lenet_model_ori.pth --model models/lenet_model.pth
