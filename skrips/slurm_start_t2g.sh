#!/bin/bash
#SBATCH --job-name=fine_tune_t2g
#SBATCH --output=fine_tunet2g.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=afkm
#SBATCH --nodelist=gpu01
#SBATCH --mem=32000


# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun -u python3 -u fine_tune_t2g.py >> Result_fine_tunet2g.txt
#srun nividia-smi