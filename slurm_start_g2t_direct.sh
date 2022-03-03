#!/bin/bash
#SBATCH --job-name=fine_tune_g2t
#SBATCH --output=fine_tune_g2t_di.txt
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
srun nvidia-smi
srun -u python3 -u direct_fine_tune_g2t.py >> Result_fine_tune_g2t_di.txt
#srun nividia-smi