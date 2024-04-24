#!/bin/bash
#SBATCH --job-name=set_env
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 48:00:00
#SBATCH -o out
#SBATCH -e err

source /home/fpesce/.bashrc

#conda remove -n PeptDes --all
conda create -n PeptDes -c pytorch -c nvidia pytorch pytorch-cuda=11.8
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
conda install -c conda-forge "ray-default" scikit-learn scikit-image opencv "ray-train" "ray-tune" "ray-rllib" gcc=12.1.0
pip install localcider==0.1.18
#pip install transformers[torch] fair-esm
