#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpu-bind
#SBATCH -w GPU19
#SBATCH -p ScoreLab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=TitanX:8
#SBATCH --gres=gpu:TitanX:8
#SBATCH --mail-user=saeed3@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=slurm_repeatability.out.%j
#SBATCH --error=slurm_repeatability_error.out.%j
srun python train_doublePrecision.py -d ../../data -iter LU3_3 -total_epochs 20 -m train

