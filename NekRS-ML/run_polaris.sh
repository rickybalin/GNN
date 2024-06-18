#!/bin/bash

module use /soft/modulefiles
module load conda/2024-04-29
conda activate /eagle/datascience/balin/SimAI-Bench/conda/clone

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=4
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

#mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1:8:16:24 ./set_affinity_gpu_polaris.sh python main.py backend=nccl halo_swap_mode=none 
#mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python main.py backend=nccl halo_swap_mode=none 2>&1 | tee out.log 
mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python main.py backend=nccl halo_swap_mode=all_to_all 2>&1 | tee out.log 
#mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python main.py backend=nccl halo_swap_mode=send_recv 2>&1 | tee out.log 




