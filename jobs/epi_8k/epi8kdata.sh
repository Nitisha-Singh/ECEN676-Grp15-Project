#!/bin/bash
#SBATCH --job-name=epi_8kdata     # create a short name for your job
#SBATCH --nodes=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=23:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=results/slurm-%x.out
#SBATCH --error=results/slurm-%x.err#!/bin/bash
#SBATCH --mail-type=END,FAIL         # Mail Events (NONE,BEGIN,FAIL,END,ALL)
#SBATCH --mail-user=nitisha@tamu.edu   # Replace with your email address

echo "champsim_epi_8k:data.gz"
bin/champsim_epi_8k \
  --warmup_instructions 200000000 \
  --simulation_instructions 1000000000 \
  /scratch/user/nitisha/Traces/data-serving.gz \
  > results/epi_8k/data.txt


