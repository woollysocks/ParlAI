#!/bin/bash

# Generic job script for all experiments.

#SBATCH --job-name=test
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -c2
#SBATCH --mem=16000
#SBATCH -t72:00:00


##############################################
# interactive info
srun -c8 -t168:00:00 --qos=interactive --pty /bin/bash 
##############################################



# Activate python environment
source parlai-env/bin/activate

# Go to correct folder
cd ~/ParlAI/parlai/mturk/tasks/game2/

# Run MTurk Job
python run.py -nc 10 -r 4.0 --live --max_connections 32 --datapath ~/Documents/General/nlp_research/data/ -sents --num_workers 2 --count_complete --allowed-conversations 1

# Enter keypress
echo -ne '\n'

deactivate




##############################################
# cost info
8min per turn
8.5 examples per hour 
3 HITs per hour => 5

3 turns per HITs
24 turns => 8 HITs => 8*5*4 = $160 = $192 w/ aws cut
##############################################

python run.py -nc 8 -r 0.0 --sandbox --max_connections 48 --datapath ~/Documents/General/nlp_research/data/ -sents --num_workers 4 --count_complete

# Live session locally
python run.py -nc 10 -r 4 --live --max_connections 48 --datapath ~/Documents/General/nlp_research/data/ -sents --num_workers 2 --count-complete --allowed-conversations 1
