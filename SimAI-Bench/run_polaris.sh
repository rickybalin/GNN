#!/bin/bash

module use /soft/modulefiles
module load conda/2024-04-29
##conda activate base
##conda activate /eagle/datascience/balin/SimAI-Bench/conda/clone
conda activate /lus/eagle/projects/datascience/balin/Nek/GNN/env/gnn
source /lus/eagle/projects/datascience/balin/Nek/GNN/env/_pyg/bin/activate

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

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



