#!/bin/bash
EXE_TO_LAUNCH=$1
if [[ -z "$EXE_TO_LAUNCH" ]]; then
    echo "You MUST specify what exe to launch!" >&2; exit -1
elif [[ -f $EXE_TO_LAUNCH && -x $EXE_TO_LAUNCH ]]; then
    echo "Will launch $(readlink -f $EXE_TO_LAUNCH)" >&2
    sbatch <<EOF
#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --qos gpu_free --gres gpu:1
#SBATCH -t 00:01:00
#SBATCH -J gpu
#SBATCH -o $(basename -s .x "$EXE_TO_LAUNCH")_gpu.out
module load gcc cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun "$EXE_TO_LAUNCH" 2>&1
EOF
else
    echo "$EXE_TO_LAUNCH not found or not a compiled exe!" >&2; exit -1
fi
