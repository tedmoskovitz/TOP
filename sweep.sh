#!/bin/bash


for seed in 104 107 110 111
do
for env in Walker2d-v2  
do
for n_quantiles in 50
do
for bandit_lr in 0.1 
do
echo "================================="
echo training seed $seed on $env with $n_quantiles quantiles, lr $lr, bandit lr $bandit_lr
sbatch --export=ALL,SEED=$seed,ENV=$env,BANDIT_LR=$bandit_lr,N_QUANTILES=$n_quantiles dope.sbatch 
echo done.
done
done
done
done
