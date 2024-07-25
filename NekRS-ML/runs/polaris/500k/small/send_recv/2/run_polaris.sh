#!/bin/bash 

module use /soft/modulefiles
module load jax/0.4.29-dev
source /lus/eagle/projects/datascience/balin/Nek/GNN/env/_pyg_old/bin/activate

echo Loaded modules:
module list
echo
echo Torch version: `python -c "import torch; print(torch.__version__)"`
echo Torch CUDA version: `python -c "import torch;print(torch.version.cuda)"`
echo NCCL version: `python -c "import torch;print(torch.cuda.nccl.version())"`
echo cuDNN version: `python -c "import torch;print(torch.backends.cudnn.version())"`
echo Torch Geometric version: `python -c "import torch;import torch_geometric;print(torch_geometric.__version__)"`
echo

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
##export NCCL_NET="AWS Libfabric"
##export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
##export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

# for all_to_all at larger scale
#export FI_CXI_RX_MATCH_MODE=software
#export FI_CXI_RDZV_PROTO=alt_read
#export FI_CXI_REQ_BUF_SIZE=8388608

NODES=$(cat $PBS_NODEFILE | wc -l)
NODES=1
PROCS_PER_NODE=2
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

# Halo swap mode
#HALO_SWAP_MODE=none
#HALO_SWAP_MODE=all_to_all
#HALO_SWAP_MODE=all_to_all_opt
HALO_SWAP_MODE=send_recv

# Data path strong scaling
#DATA_PATH=/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/gnn_outputs_distributed_gnn/gnn_outputs_poly_3/

# Data path weak scaling
#DATA_PATH=/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv_weak_scaling/ne_16_v2/gnn_outputs_poly_5/
DATA_PATH=/eagle/datascience/balin/Nek/GNN/weak_scale_data/500k_polaris/${PROCS}/gnn_outputs_poly_5/

EXE=/eagle/projects/datascience/balin/Nek/GNN/GNN/NekRS-ML/main.py
ARGS="hidden_channels=8 n_mlp_hidden_layers=2 backend=nccl halo_swap_mode=${HALO_SWAP_MODE} gnn_outputs_path=${DATA_PATH}"
echo Running script $EXE
echo with arguments $ARGS
echo
echo `date`
mpiexec --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python $EXE ${ARGS} 
echo `date`



