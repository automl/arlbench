#!/bin/bash
module load tools Apptainer
mkdir /dev/shm/$2 -p

export SINGULARITY_BIND=/scratch/$1/$2:/tmp/scratch

export SINGULARITY_CACHEDIR=$PC2PFS/$1/$2/SINGULARITY_CACHE
export SINGULARITY_TMPDIR=/dev/shm/$2

export APPTAINER_CACHEDIR=$PC2PFS/$1/$2/SINGULARITY_CACHE
export APPTAINER_TMPDIR=/dev/shm/$2

export TMPDIR=$PC2PFS/$1/$2/TEMPDIR
export TEMP=$PC2PFS/$1/$2/TEMPDIR
export TMP=$PC2PFS/$1/$2/TEMPDIR

apptainer build singularity_container.sif singularity_container.recipe
