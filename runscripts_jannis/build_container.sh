#!/bin/bash
module load Singularity/3.5.3
mkdir $BIGWORK/SINGULARITY_CACHE -p
mkdir $BIGWORK/TMPDIR -p

export SINGULARITY_CACHEDIR=$BIGWORK/SINGULARITY_CACHE

export TMPDIR=$BIGWORK/TMPDIR
export TEMP=$BIGWORK/TMPDIR
export TMP=$BIGWORK/TMPDIR

singularity build --fakeroot --bind /opt/software/slurm,/var/run/munge,/run/munge,/usr/lib64/libmunge.so.2,/usr/lib64/libmunge.so.2.0.0,$BIGWORK:/tmp/bigwork singularity_container.sif singularity_container.recipe