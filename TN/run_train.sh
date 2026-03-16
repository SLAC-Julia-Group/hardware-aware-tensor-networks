#!/bin/bash -l
#SBATCH -A m2616           # e.g. this needs to be your project account
#SBATCH --qos shared             # the queue name is slightly different if using shares use --qos 
#SBATCH -t 30:00:00                 # walltime ==> smaller time limit faster to get resources
#SBATCH -N 1                        # one node
#SBATCH -C gpu                      # request GPU nodes
#SBATCH --gpus-per-node=1           # one A100 GPU
#SBATCH --ntasks=1                  # one task
#SBATCH --cpus-per-task=32           
#SBATCH --job-name=Rearranged_SMPO_Nominal_Ensemble_Seeds_10_31415
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Good defaults for PyTorch on Perlmutter
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_GPU_BIND=closest
export PYTHONUNBUFFERED=1

module load conda
conda activate qiml_prajita3_copy  # <-- change to your env
python -c "import jax; print(jax.devices())"

# Move to your project dir (edit this)
cd /global/homes/p/prajitab/QIML_HLS/QIML_Fresh/QiML/TN/

EPOCHS=200
BATCH=2048
## ==>> to run just a single training, use following command
#randomly 
# 13
# 42
# 97
# 123
# 256
# 512
# 777
# 1024
# 2025
# 31415

srun -u python training_script.py --seed 31415 --output_folder "../output/Nominal_SMPO_Sagar_Seed31415_Epoch200/" --epochs ${EPOCHS} --batch_size ${BATCH}

srun -u python training_script.py --seed 512 --output_folder "../output/Nominal_SMPO_Sagar_Seed512_Epoch200/" --epochs ${EPOCHS} --batch_size ${BATCH}

## similarly add more runs with different seeds as needed