#!/bin/bash
#SBATCH --nodes=1
#SBATCH -w GPU8
#SBATCH -p Contributors
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mail-user=saeed3@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=slurm_output.out.%j
#SBATCH --error=slurm_error.out.%j
#srun python -u ./pytorch_code/code/train_doublePrecision.py -d ./data -GPU 12 -m test -iter LU3-OnFlyAug_14 
srun python -u ./pytorch_code/code/train_doublePrecision.py -d ./data -GPU None -m train -iter 1 
