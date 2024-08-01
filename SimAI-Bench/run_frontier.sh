#!/bin/bash

module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/5.7.1
module load craype-accel-amd-gfx90a
source activate /lustre/orion/csc613/proj-shared/balin/Nek/GNN/env/gnn

# Get address of head node
export MASTER_ADDR=$(hostname -i)

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1

NODES=$(echo $SLURM_NODEFILE | wc -l)
PROCS_PER_NODE=8
PROCS=$((NODES * PROCS_PER_NODE))
echo Number of nodes: $NODES
echo Number of ranks per node: $PROCS_PER_NODE
echo Number of total ranks: $PROCS
echo

srun -N$NODES -n$PROCS --ntasks-per-node=$PROCS_PER_NODE -c7 --gpu-bind=closest \
    python main.py --device=cuda --iterations=20 --problem_size=large \
    --master_addr=$MASTER_ADDR --master_port=3442



