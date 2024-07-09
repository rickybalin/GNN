#!/bin/bash

module use /soft/modulefiles
module load conda/2024-04-29
#conda activate /eagle/datascience/balin/SimAI-Bench/conda/clone
conda activate /lus/eagle/projects/datascience/balin/Nek/GNN/env/gnn
source /lus/eagle/projects/datascience/balin/Nek/GNN/env/_pyg/bin/activate

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
##export NCCL_NET="AWS Libfabric"
##export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
##export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
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

# Halo swap mode
#HALO_SWAP_MODE=none
#HALO_SWAP_MODE=all_to_all
HALO_SWAP_MODE=all_to_all_opt
#HALO_SWAP_MODE=send_recv

# Data path strong scaling
#DATA_PATH=/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/gnn_outputs_distributed_gnn/gnn_outputs_poly_3/

# Data path weak scaling
#DATA_PATH=/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv_weak_scaling/ne_128_v2/gnn_outputs_poly_5/
DATA_PATH=/eagle/datascience/balin/Nek/GNN/weak_scale_data/500k_polaris/${PROCS}/gnn_outputs_poly_5/

#mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1:8:16:24 ./set_affinity_gpu_polaris.sh python main.py backend=nccl halo_swap_mode=none 
mpiexec --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 \
    python main.py \
    backend=nccl \
    halo_swap_mode=$HALO_SWAP_MODE \
    gnn_outputs_path=$DATA_PATH \
    2>&1 | tee out.log 




