#!/bin/bash
module load tools Apptainer
mkdir /dev/shm/$1 -p

export SINGULARITY_BIND=/scratch/hpc-prf-intexml/$1:/tmp/scratch

export SINGULARITY_CACHEDIR=$PC2PFS/hpc-prf-intexml/$1/SINGULARITY_CACHE
export SINGULARITY_TMPDIR=/dev/shm/$1

export APPTAINER_CACHEDIR=$PC2PFS/hpc-prf-intexml/$1/SINGULARITY_CACHE
export APPTAINER_TMPDIR=/dev/shm/$1

apptainer build singularity_container.sif singularity_container.recipe
