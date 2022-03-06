#!/bin/bash
#SBATCH --job-name=fine_tune_g2t
#SBATCH --output=fine_tune_g2t.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=afkm
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:1
#SBATCH --mem=32000


# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun -u python3 -u fine_tune_g2t.py >> Result_fine_tune_g2t.txt
#srun nividia-smi