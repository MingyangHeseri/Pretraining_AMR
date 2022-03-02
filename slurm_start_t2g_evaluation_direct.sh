#!/bin/bash
#SBATCH --job-name=t2g_eval
#SBATCH --output=t2g_eval_di.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=afkm
#SBATCH --nodelist=gpu01
#SBATCH --mem=64000


# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun -u python3 -u evaluation_t2g_direct.py >> Result_eval_tunet2g_di.txt
#srun nividia-smi