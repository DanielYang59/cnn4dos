#!/bin/bash 
#PBS -N test_gputraining
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=1GB
#PBS -l walltime=0:30:00
#PBS -q gpuvolta
#PBS -P iu57
#PBS -l storage=gdata/iu57+gdata/v88+gdata/dk92
#PBS -v PYTHONPATH=/g/data/v88/EXTRA_PYTHON_LIB/lib/python3.9/site-packages

## Activate TensorFlow environment
module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/22.11

# Run TensorFlow
cd $PBS_O_WORKDIR
export TF_GPU_THREAD_MODE="gpu_private"
python3 training.py > tunerlog 2>&1
