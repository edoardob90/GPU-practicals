#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p debug
#SBATCH -t 00:01:00
#SBATCH -J omp
#SBATCH -o task4_omp_8.out

module load gcc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun --hint=nomultithread ./task4_4096_omp.x 2>&1
