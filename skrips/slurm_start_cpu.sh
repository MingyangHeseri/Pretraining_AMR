#!/bin/bash
#SBATCH --job-name=input_pre_show
#SBATCH --output=input_pre_show.txt
#SBATCH --qos=batch
#SBATCH --mail-user=he@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=compute
#SBATCH --nodelist=node39




# Add ICL-Slurm binaries to path
#PATH=/opt/slurm/bin:$PATH

## JOB STEPS
srun python3 pretrain_bart.py >> Result_input_show1.txt
#srun nividia-smi