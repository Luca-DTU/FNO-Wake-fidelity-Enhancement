#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J fno-wake-test
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### ask for gpu
### BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 8GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 400
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s220647@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute
# module load cuda/11.7
# ~/miniconda3/envs/hpc_env/bin/python main.py
# python main.py -m +experiment=grid_search hydra/launcher=joblib
# python main.py --multirun
~/miniconda3/envs/thesis/bin/python main.py hydra/launcher=joblib