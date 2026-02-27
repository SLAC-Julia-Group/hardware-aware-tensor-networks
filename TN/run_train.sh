#!/bin/bash -l
#SBATCH -A m2616           # e.g. this needs to be your project account
#SBATCH --qos shared             # the queue name is slightly different if using shares use --qos 
#SBATCH -t 4:00:00                 # walltime ==> smaller time limit faster to get resources
#SBATCH -N 1                        # one node
#SBATCH -C gpu                      # request GPU nodes
#SBATCH --gpus-per-node=1           # one A100 GPU
#SBATCH --ntasks=1                  # one task
#SBATCH --cpus-per-task=32           
#SBATCH --job-name=CSMPO_8_3_42abc
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Good defaults for PyTorch on Perlmutter
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_GPU_BIND=closest
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate qiml_prajita3  # <-- change to your env

# Move to your project dir (edit this)
cd /global/homes/p/prajitab/QIML_HLS/QiML/TN/

OUTROOT=../output/Nominal_Ensemble_CSMPO_BondDim8_BatchSize4096_Epoch200_seed38/
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

srun -u python training_script.py --seed 42 --output_folder "../output/Nominal_SMPO_Sagar_Seed42/" --epochs ${EPOCHS} --batch_size ${BATCH}

# srun -u python training_script.py --seed 42 --output_folder "../output/Nominal_Ensemble_CSMPO_BondDim8_BatchSize4096_Epoch200_seed42b/" --epochs ${EPOCHS} --batch_size ${BATCH}

# srun -u python training_script.py --seed 42 --output_folder "../output/Nominal_Ensemble_CSMPO_BondDim8_BatchSize4096_Epoch200_seed42c/" --epochs ${EPOCHS} --batch_size ${BATCH}


# # Loop 20 times to run 20 independent experiments
# for i in 44 46 49 52 54 56 58 60 62 64; do 
#     # Generate a high-quality random seed (0 to 4 billion), simply iterating is generally not recommended for randomness
#     SEED=i#$(shuf -i 0-4294967295 -n 1)
    
#     OUTPUTFOLDER=SMPO_Nominal_Ensemble_Seed${SEED}_Epochs${EPOCHS}_Batch${BATCH}
#     mkdir -p "${OUTROOT}/${OUTPUTFOLDER}"
    
#     echo "Launching experiment $i with random seed: $SEED"
    
#     # Run the script
#     srun -u python training_script.py \
#         --seed "${SEED}" \
#         --output_folder "${OUTROOT}/${OUTPUTFOLDER}" \
#         --epochs "${EPOCHS}" \
#         --batch_size "${BATCH}"
# done
