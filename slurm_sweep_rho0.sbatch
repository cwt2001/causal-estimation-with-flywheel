#!/bin/bash
#SBATCH --job-name=aa  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node (adjust when using MPI)
#SBATCH --cpus-per-task=100     # cpu-cores per task (>1 if multi-threaded tasks, adjust when using OMP)
#SBATCH --time=2-24:00:00          # total run time limit (D-HH:MM:SS)
#SBATCH --partition=amd        # partition(queue) where you submit (amd/intel/gpu-a30/gpu-l20)
#SBATCH --account=group        # slurm account name
#SBATCH --output=results/slurm_%j.out    # output file name
#SBATCH --error=results/slurm_%j.err     # error file name

python run_rho0_scaling_experiments.py --config configs.config --n 5000 --m-edges 20 --max-workers 100 --targets 0.05,0.2,0.5,0.8
# sbatch --wait -o slurm.out slurm_sweep_rho0.sbatch
