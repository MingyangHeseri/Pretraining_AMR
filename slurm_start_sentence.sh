#!/bin/bash
#SBATCH --job-name=pretrain_sentence
#SBATCH --output=pre_train_bart_sentence.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=afkm
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu01
#SBATCH --mem=16000


# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun -u python3 -u pre_train_sentence.py >> Result_pre_train_bart_sentence.txt
#srun nividia-smi