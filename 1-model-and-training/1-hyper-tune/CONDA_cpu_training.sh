#!/bin/bash 
#PBS -N hypertune-CPU
#PBS -l ncpus=48
#PBS -l mem=168GB
#PBS -l jobfs=1GB
#PBS -l walltime=10:00:00
#PBS -q normal
#PBS -P iu57
#PBS -l storage=gdata/iu57+gdata/v88+gdata/dk92

cd $PBS_O_WORKDIR

# Activate conda environment
eval "$(conda shell.bash hook)"  # https://hpc-unibe-ch.github.io/software/Anaconda.html
conda activate /g/data/v88/CONDA_ENV/tensorflow

# Run Horovod
python3 keras-tuner.py > tunerlog 2>&1
