#!/bin/bash
#SBATCH --job-name=evalmask_sentence_g2t
#SBATCH --output=eval_mask_sentence.txt
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
srun -u python3 -u evaluation_masked_sentences.py >> Result_masked_sentences.txt
#srun nividia-smi