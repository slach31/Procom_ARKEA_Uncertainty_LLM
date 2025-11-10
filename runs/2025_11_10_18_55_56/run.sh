#!/bin/bash
#SBATCH --job-name=2025_11_10_18_55_56
#SBATCH --partition=Global
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-gpu=8
#SBATCH --output="/Brain/public/projects/projetArkea/Procom_ARKEA_Uncertainty_LLM/runs/2025_11_10_18_55_56/stdout.log"
#SBATCH --error="/Brain/public/projects/projetArkea/Procom_ARKEA_Uncertainty_LLM/runs/2025_11_10_18_55_56/stderr.log"
#SBATCH --chdir="/Brain/public/projects/projetArkea/Procom_ARKEA_Uncertainty_LLM/src"
export JOB_DIR="/Brain/public/projects/projetArkea/Procom_ARKEA_Uncertainty_LLM/runs/2025_11_10_18_55_56"
export WANDB_DIR="/Brain/public/projects/projetArkea/Procom_ARKEA_Uncertainty_LLM/runs/2025_11_10_18_55_56/wandb"
export HF_HOME="$SCRATCH/s23lachg/caches/hf_home"
export MPLCONFIGDIR="$SCRATCH/s23lachg/caches/matplotlib"
export XDG_CACHE_HOME="$SCRATCH/s23lachg/caches/yt_dlp"
srun python main.py
