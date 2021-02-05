#!/bin/bash
#SBATCH --job-name=dope
#SBATCH --output=./logs/dope_%A.out
#SBATCH --time=0-16:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gres gpu:1

cd
source pop/bin/activate
cd ~/DOPE

module load cuda/10.1


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/ghome/live/theodorm/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-430

srun python3.6 train_dope_agent.py --n_quantiles=$N_QUANTILES --seed=$SEED --bandit_lr=$BANDIT_LR --env=$ENV 
