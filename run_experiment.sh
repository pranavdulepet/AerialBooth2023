#!/bin/bash

#SBATCH --job-name=diffusion_exp
#SBATCH --output=diffusion_exp_%j.out
#SBATCH --error=diffusion_exp_%j.err
#SBATCH --time=24:00:00
#SBATCH --pty --gres=gpu:8 --account=gamma --partition=gamma --qos=huge-long --mem=248G
#SBATCH --mem=32G

module load python/3.11 

source ../.venv/bin/activate

python3 experiment_pipeline_noisy.py

