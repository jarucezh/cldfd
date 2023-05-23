#!/bin/bash
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=24:00:00
#$ -p -3
#$ -N student96_euro_data2_coef0001
source ~/anaconda3/bin/activate dac
module load cuda

python oho.py --dir tmp/student96_euro_data2_coef0001 --target_dataset EuroSAT --target_subset_split splits/EuroSAT_2_unlabeled.csv --alpha 2 --bsize 32 --coef 0.001

