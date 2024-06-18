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

#mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python main.py --device=cuda --iterations=100 --problem_size=medium
mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python main.py --device=cuda --iterations=10 --problem_size=large



