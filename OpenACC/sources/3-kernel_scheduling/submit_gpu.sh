#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --qos gpu_free --gres gpu:1
#SBATCH -t 00:01:00
#SBATCH -J gpu
#SBATCH -o task4_gpu_2.out

module load gcc cuda

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PGI_ACC_TIME=1
srun --hint=nomultithread ./task4_out.x 2>&1
