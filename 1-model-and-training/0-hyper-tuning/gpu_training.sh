#!/bin/bash 
#PBS -N tuner
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=192GB
#PBS -l jobfs=1GB
#PBS -l walltime=12:00:00
#PBS -q gpuvolta
#PBS -P v88
#PBS -l storage=gdata/iu57+gdata/v88+gdata/dk92

cd $PBS_O_WORKDIR

# Request interactive GPU
# qsub -I -q gpuvolta -lwd,walltime=1:00:00,ngpus=1,ncpus=12,mem=100GB,jobfs=1GB,storage=gdata/iu57+gdata/v88+gdata/dk92


# Activate conda environment
# eval "$(conda shell.bash hook)"  # https://hpc-unibe-ch.github.io/software/Anaconda.html
conda activate /g/data/v88/CONDA_ENV/tensorflow


# Run Horovod
python3 keras-tuner.py > tunerlog 2>&1


# GPU monitoring
# Ref: https://opus.nci.org.au/display/DAE/Gpustat+-+Multinode+GPU+monitoring
# module load gpustat/1.0
# gpustat-run <job_no>.gadi-pbs
