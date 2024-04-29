# Use Singularity to run ARLBench on PC2

1. To build the container, insert your group and user names and run the following command:
   **Important**: The user name has to match the directory name in your group's directory in /scratch!

```bash
./build_container.sh <GROUP> <USER_NAME>
```

2. To run a script run:

```bash
sbatch run.sh
```
