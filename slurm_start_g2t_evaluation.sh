#!/bin/bash
#SBATCH --job-name=evalg2t_g2t
#SBATCH --output=evalg2t.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=afkm
#SBATCH --nodelist=gpu01
#SBATCH --mem=32000
#SBATCH --gres=gpu:1


# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun -u python3 -u evaluation_g2t.py >> Result_evalg2t.txt
#srun nividia-smi