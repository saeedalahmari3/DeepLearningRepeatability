#!/bin/bash
#SBATCH --nodes=1
#SBATCH -w GPU5
#SBATCH -p Extended
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1080Ti:1
#SBATCH --mail-user=saeed3@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=slurm_repeatability.out.%j
#SBATCH --error=slurm_repeatability_error.out.%j
srun python train_singlePrecision.py -d ../../data -iter LU3_1 -total_epochs 100 -m test -GPU 5
srun python train_singlePrecision.py -d ../../data -iter LU3_2 -total_epochs 100 -m test -GPU 5
srun python train_singlePrecision.py -d ../../data -iter LU3_3 -total_epochs 100 -m test -GPU 5
srun python train_singlePrecision.py -d ../../data -iter LU3_4 -total_epochs 100 -m test -GPU 5
srun python train_singlePrecision.py -d ../../data -iter LU3_5 -total_epochs 100 -m test -GPU 5
srun python train_singlePrecision.py -d ../../data -iter LU3_6 -total_epochs 100 -m test -GPU 5
srun python train_singlePrecision.py -d ../../data -iter LU3_7 -total_epochs 100 -m test -GPU 5
