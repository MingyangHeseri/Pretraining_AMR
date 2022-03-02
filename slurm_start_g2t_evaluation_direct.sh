#!/bin/bash
#SBATCH --job-name=eva_g2t_di
#SBATCH --output=eva_g2t_di.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=afkm
#SBATCH --nodelist=gpu01
#SBATCH --mem=32000


# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun -u python3 -u evaluation_g2t_direct.py >> Result_eva_g2t_di.txt
#srun nividia-smi