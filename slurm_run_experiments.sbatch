#!/bin/bash
#SBATCH --job-name=experiments  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node (adjust when using MPI)
#SBATCH --cpus-per-task=100      # cpu-cores per task (>1 if multi-threaded tasks, adjust when using OMP)
#SBATCH --time=4-24:00:00          # total run time limit (D-HH:MM:SS)
#SBATCH --partition=amd        # partition(queue) where you submit (amd/intel/gpu-a30/gpu-l20)
#SBATCH --account=group        # slurm account name
#SBATCH --output=results/experiments_%j.out    # output file name
#SBATCH --error=results/experiments_%j.err     # error file name

# Forward all sbatch script arguments to experiments.py.
# Example:
# sbatch --wait --export=ALL,USE_BAYESIAN_BGPS=1 -o slurm.out slurm_run_experiments.sbatch \
#   --setting configs.experiment_settings.bgps_setting --config configs.exp_config

python experiments.py "$@"
