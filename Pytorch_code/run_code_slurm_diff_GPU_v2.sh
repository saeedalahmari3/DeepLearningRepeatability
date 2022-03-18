#!/bin/bash
#SBATCH --nodes=1
#SBATCH -w GPU19
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:TitanX:1
#SBATCH --mail-user=saeed3@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=slurm_repeatability.out.%j
#SBATCH --error=slurm_repeatability_error.out.%j
srun python train_doublePrecision2.py -d ../../data -iter LU3-NoWeightDecay-GPU19_7 -total_epochs 20 -m train -GPU 19

