#!/bin/bash
#SBATCH --job-name=mask_evaluate
#SBATCH --output=mask_evaluation.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=afkm
#SBATCH --nodelist=gpu01
#SBATCH --mem=16000
#SBATCH --gres=gpu:1


# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun -u python3 -u evaluation_masked_model.py >> evaluation_masked_model.txt
#srun nividia-smi