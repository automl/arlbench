# pc2_slurm_cluster_template
This repository is a template featuring elements to run Python experiments on the PC^2 HPC system (cluster) at Paderborn University, called Noctua2. Noctua2 uses SLURM as a job management system.

Noctua2 generally does not allow you to install software packages directly. You can see available packages [here](https://uni-paderborn.atlassian.net/wiki/spaces/PC2DOK/pages/14024771/All+Software+Modules); keep in mind that most of them need to be loaded manually as per the [documentation](https://uni-paderborn.atlassian.net/wiki/spaces/PC2DOK/pages/13634213/Loading+Software+Environments+Using+Modules). The easiest way around this limitation is to use [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) containers (similar to Docker containers) and run your code inside a Singularity container. To this end, we will use several bash scripts, some of which are used to define the properties of your SLURM job and some which define the code to be run inside the Singularity container. 

Note: If you only need python packages, it is now possible to install python packages directly; see the [documentation](https://uni-paderborn.atlassian.net/wiki/spaces/PC2DOK/pages/15335425/Python) on how to set this up. Furthermore, is is possible to install software in the project directory manually (copying files).

## General Flow - Singularity
1. Create a requirements.txt with all packages required to run your code.
2. Adapt the Singularity container recipe called `singularity_container.recipe` to install anything additionally on the Ubuntu system inside to use. A recipe is essentially a build plan for the container. For more details refer to the [documentation](https://docs.sylabs.io/guides/3.5/user-guide/definition_files.html).
3. Build your Singularity container on a Linux machine, where you have admin rights. You CANNOT do this on the cluster itself due to missing permissions.

    ``sudo singularity build singularity_container.sif singularity_container.recipe``
4. Resolve any problems occuring while building the container until your have a Singularity container built from your recipe.
5. Edit the `run_in_container.sh` to call your code. Everything inside this file will be executed inside the Singularity container when the corresponding job is allocated.
6. Define the cluster job parameters at the top of the `run.sh` file. 
7. Upload everything onto the cluster and submit your job via

    ``sbatch run.sh``
8. Check the status of your job via one of the following commands: 
    * `squeue_pretty`
    * `spredict` (to get a prediction of the starttime of your job) 

## Caveat
Note that this repository only holds samples and not a complete documentation. There are several resource, if you are looking for a complete documentation: 
* [PC^2 Wiki](https://uni-paderborn.atlassian.net/wiki/spaces/PC2DOK/overview?homepageId=12943374)
* [CECI-HPC Slurm Wiki](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html#)

