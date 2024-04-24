#!/bin/bash
#SBATCH --job-name=ppo_tuner
#SBATCH --partition=qgpu
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:2
#SBATCH -t 50:00:00
#SBATCH -o out
#SBATCH -e err

source /home/fpesce/.bashrc

conda activate PeptDes

python ppo_tuner_TD.py
