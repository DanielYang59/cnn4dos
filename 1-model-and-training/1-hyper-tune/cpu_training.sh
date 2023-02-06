#!/bin/bash 
#PBS -N hypertune-CPU
#PBS -l ncpus=48
#PBS -l mem=168GB
#PBS -l jobfs=1GB
#PBS -l walltime=24:00:00
#PBS -q normal
#PBS -P iu57
#PBS -l storage=gdata/iu57+gdata/v88+gdata/dk92
#PBS -v PYTHONPATH=/g/data/v88/EXTRA_PYTHON_LIB/lib/python3.9/site-packages

cd $PBS_O_WORKDIR

# Activate tensorflow environment
module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/22.11

# Run Horovod
python3 keras-tuner.py > tunerlog 2>&1
