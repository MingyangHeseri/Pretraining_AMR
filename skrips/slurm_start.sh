#!/bin/bash
#SBATCH --job-name=pretrain_bart
#SBATCH --output=pre_train_bart1.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=afkm
#SBATCH --nodelist=gpu01
#SBATCH --mem=64000


# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun -u python3 -u pretrain_bart.py >> Result_pre_train_bart.txt
#srun nividia-smi