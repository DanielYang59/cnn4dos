#!/bin/bash 
#PBS -N hypertune-GPU-Gadi-env
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=168GB
#PBS -l jobfs=1GB
#PBS -l walltime=1:00:00
#PBS -q gpuvolta
#PBS -P iu57
#PBS -l storage=gdata/iu57+gdata/v88+gdata/dk92
#PBS -v PYTHONPATH=/g/data/v88/EXTRA_PYTHON_LIB/lib/python3.9/site-packages

cd $PBS_O_WORKDIR

# Request interactive GPU
# qsub -I -q gpuvolta -lwd,walltime=1:00:00,ngpus=4,ncpus=48,mem=168GB,jobfs=1GB,storage=gdata/iu57+gdata/v88+gdata/dk92


## Activate conda environment
module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/22.11


# Run Horovod
python3 keras-tuner.py > tunerlog 2>&1


# GPU monitoring
# Ref: https://opus.nci.org.au/display/DAE/Gpustat+-+Multinode+GPU+monitoring
# module load gpustat/1.0
# gpustat-run <job_no>.gadi-pbs
