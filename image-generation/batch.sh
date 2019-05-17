#!/bin/bash
#SBATCH --time=02-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=pokemon
#SBATCH --mem=16000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.pentaliotis@student.rug.nl

module load Python/3.6.4-foss-2018a
module load cuDNN/7.1.4.18-CUDA-9.0.176

python -u train.py
