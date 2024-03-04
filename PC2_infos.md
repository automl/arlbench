# Working on PC2
You'll need a PC2 account as well as VPN to the cluster to work on PC2. For information on how to setup the access to the PC2 cluster, please check the [our quickstart guide](https://docs.google.com/document/d/1BavfcqX6hdbElOb3xKcTOLKhutgzKWCfk1WIh1lh12Y/edit?usp=sharing). You can skip setting up the paths, we'll do that for you, you only need access.

PC2 also has a good [documentation](https://upb-pc2.atlassian.net/wiki/spaces/PC2DOK/overview) you should take a look at! 

## Queues & Resources
We have access to several queues on PC2 which are split into resources. Please note that we have limits on how much we can run, so if you need a lot of resources, check in with the PC2 Mattermost channel first.

These queues exist on noctua2: 

| Queue    | Nodes | Cores per node  | GPUs                                           | Timelimit            | Comments    |
| -------- | ----- | --------------- | ---------------------------------------------- | -------------------- | ----------- |
| normal   | 990   | 128             | -                                              | 21d                  |             |
| large_mem| 66    | 128             | -                                              | 21d                  |             |
| huge_mem | 5     | 128             | -                                              | 21d                  |             |
| gpu      | 32    | 128             | 4 A100 @ 40GB                                  | 7d                   |             |
| dgx      | 1     | 128             | 8 A100 @ 40GB                                  | resource dependent   | for testing | 
| fpga     | 35    | 128             | 16x 3 Xilinx Alveo U280, 16x 2 Bittware 520N   | 7d                   |             |

As for disk resources, you have three directories: $HOME, $SCRATCH and $DATA. It's recommended to stay away from $HOME since it's the smallest and instead work on $SCRATCH. $DATA can then be used for data storage. `pc2status` can give you more information about the status of these directories.

## Package Management
On PC2, you can in principle work with all installed global packages (including conda) or use user installs, e.g. in the form of conda environments or singularity containers.

We'll set up your conda with the correct paths if you choose so, for singularity you have an example of how to use containers (check the `README.md`). In both cases, take care to not store your pacakges in your $HOME directory - it's very small and not will likely be filled quickly. Following the singularity example and our conda shortcuts will prevent this, so take care to use our aliases `conda-create` and `conda-activate` everywhere and study the singularity example carefully.

## Interactive Jobs
If you want to debug something on the cluster or monitor a job manually, you can use an interactive job to run your commands on a node directly. You can request resources in the same way as you would do in a slurm bash script:
```bash
salloc --time=1:00:00 --nodes=1 --ntasks=16 --mem-per-cpu=4G
```

## Synching your Data
You can use rsync for synchronization. For synching from the cluster to your local machine, use:
```bash
# Cluster to local
rsync2local <your-data-path> <local-path>
```
which translates to `alias rsyn2local="rsync -azv --delete -e 'ssh -J $USER@fe.noctua2.pc2.uni-paderborn.de' $USER@n2login5:$REPODIR/$1 $2"`.

You can use the same command in reverse to synch from local to cluster, but you can't use any of the cluster shortcuts. That means you should use:
```bash
alias rsyn2cluster="rsync -azv --delete -e 'ssh -J intexml8@fe.noctua2.pc2.uni-paderborn.de' <local-path> intexml8@n2login5:/scratch/hpc-prf-intexml/teimer/projects/<remote-path>"
```
Instead of:
```bash
alias rsyn2cluster="rsync -azv --delete -e 'ssh -J $USER@fe.noctua2.pc2.uni-paderborn.de' <local-path> $USER@n2login5:$REPODIR/<remote-path>"
```
We recommend making a local bash alias for this.

The commands mean:
- a: archive mode, copies recursively and keeps permissions etc
- z: compress file data during the transfer
- v: increase verbosity
- delete: delete extraneous files from dest dirs # TODO[Caro] Do we want this? I find it useful. 

## Permissions
Your have access and can work in our data and project directories - that means we have access to each others' data. Please take care to only access others' directories when your collaborating and have explicit permission to do so.

## Debugging on the cluster / Using the cluster as a code server
If you use Visual Studio Code, you can remote into the cluster.
For this you need to setup an extra ssh config to properly configure the ssh proxy jumps.
You can find the necessary ssh configs [here](https://upb-pc2.atlassian.net/wiki/spaces/PC2DOK/pages/1902225/Access+for+Applications+like+Visual+Studio+Code).

## Best Practices
- Use $HOME as little as possible
- Check your quotas regularly
- Don't reserve whole nodes if you don't actually need them 
