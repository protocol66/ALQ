#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --mincpus=8
#SBATCH --partition=gpu-warp
#SBATCH --output=./logs/slurm/slurm-%j-vgg.out
. /opt/ohpc/pub/spack/0.18.1/share/spack/setup-env.sh
spack load miniconda3 cudnn@8.2.4 cuda@11.7
source activate Tensorflow
python vgg.py --PRETRAIN --DOWNLOAD --ALQ --POSTTRAIN --data ~/Datasets/cifar10 --model_ori models/vgg_small_model_ori.pth --model models/vgg_small_model.pth
