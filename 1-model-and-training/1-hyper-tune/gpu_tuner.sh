#!/bin/bash --login
#PBS -N z-keras-tuner
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=96GB
#PBS -l jobfs=50GB
#PBS -l walltime=24:00:00
#PBS -q gpuvolta
#PBS -P iu57
#PBS -l storage=gdata/iu57+gdata/v88+gdata/dk92
#PBS -v PYTHONPATH=/g/data/v88/EXTRA_PYTHON_LIB/lib/python3.9/site-packages


## Activate TensorFlow environment
module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/22.11


# Run Keras tuner
cd $PBS_O_WORKDIR
python3 keras-tuner.py > tunerlog 2>&1


# Request interactive GPU
# qsub -I -q gpuvolta -lwd,walltime=1:00:00,ngpus=4,ncpus=48,mem=168GB,jobfs=1GB,storage=gdata/iu57+gdata/v88+gdata/dk92

# GPU monitoring
# Ref: https://opus.nci.org.au/display/DAE/Gpustat+-+Multinode+GPU+monitoring

