#!/bin/bash
#SBATCH --job-name=pretrain_bart_subgraph
#SBATCH --output=pre_train_bart_subgraph.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=afkm
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:1
#SBATCH --mem=16000



# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun -u python3 -u pretrain_bart_subgraph.py >> Result_pre_train_bart_subgraph.txt
#srun nividia-smi