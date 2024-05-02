#!/bin/bash
module load tools Apptainer
mkdir /dev/shm/jbecktepe -p

apptainer build singularity_container.sif singularity_container.recipe
