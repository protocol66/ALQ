#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --mincpus=8
## SBATCH --partition=gpu-warp
#SBATCH --output=./logs/slurm/slurm-%j-res18.out
. /opt/ohpc/pub/spack/0.18.1/share/spack/setup-env.sh
spack load miniconda3 cudnn@8.2.4 cuda@11.7
source activate Tensorflow
python resnet.py --PRETRAIN --DOWNLOAD --ALQ --POSTTRAIN --net resnet18 --data ~/Datasets/imagenet --batch_size 128 --model_ori models/resnet18_model_ori.pth --model models/resnet18_model.pth
