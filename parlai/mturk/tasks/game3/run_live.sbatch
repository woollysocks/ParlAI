#!/bin/bash

#SBATCH --job-name=mturk
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -c4
#SBATCH --mem=8000
#SBATCH -t72:00:00

source /home/nn1119/.bash_profile
source /home/nn1119/py3.6.3/bin/activate

python3 run.py -nc 40 -r 4.0 --sandbox --max_connections 48 --datapath /home/nn1119/data/ -sents --num_workers 2 --count_complete --allowed-conversations 1

