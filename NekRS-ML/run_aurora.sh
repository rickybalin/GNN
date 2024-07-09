#!/bin/bash

module load frameworks/2024.1
source /lus/flare/projects/Aurora_deployment/balin/Nek/GNN/env/_pyg/bin/activate

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=12
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20

export CPU_BIND="verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96"
#export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"

# CCL backend
CCL_BACKEND=ccl

# Halo swap mode
#HALO_SWAP_MODE=none
HALO_SWAP_MODE=all_to_all
#HALO_SWAP_MODE=all_to_all_opt
#HALO_SWAP_MODE=send_recv

# Data path weak scaling
DATA_PATH=/flare/Aurora_deployment/balin/Nek/GNN/weak_scale_data/500k_aurora/${PROCS}/gnn_outputs_poly_5/

mpiexec --pmi=pmix --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=${CPU_BIND} \
    python main.py \
    epochs=25 \
    backend=$CCL_BACKEND \
    halo_swap_mode=$HALO_SWAP_MODE \
    gnn_outputs_path=${DATA_PATH} \
    2>&1 | tee out.log 




