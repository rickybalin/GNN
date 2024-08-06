#!/bin/bash
#SBATCH -N 8
#SBATCH -A CSC613
#SBATCH -J gnn_scale
#SBATCH -o gnn_scale.o%j
##SBATCH -e gnn_scale-%j.e
#SBATCH -t 00:30:00
#SBATCH -p batch

cd $SLURM_SUBMIT_DIR

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/5.7.1
module load craype-accel-amd-gfx90a
source activate /lustre/orion/csc613/proj-shared/balin/Nek/GNN/env/gnn

echo Loaded modules:
module list
echo
echo Torch version: `python -c "import torch; print(torch.__version__)"`
echo NCCL version: `python -c "import torch;print(torch.cuda.nccl.version())"`
echo cuDNN version: `python -c "import torch;print(torch.backends.cudnn.version())"`
echo Torch Geometric version: `python -c "import torch;import torch_geometric;print(torch_geometric.__version__)"`
echo

# Get address of head node
export MASTER_ADDR=$(hostname -i)

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# AWS plugin
export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/sajal/software/aws-ofi-rccl/src/.libs/:${LD_LIBRARY_PATH}

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1

NODES=$(echo $SLURM_NNODES)
PROCS_PER_NODE=8
PROCS=$((NODES * PROCS_PER_NODE))
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

# Halo swap mode
HALO_SWAP_MODE=none
#HALO_SWAP_MODE=all_to_all
#HALO_SWAP_MODE=all_to_all_opt
#HALO_SWAP_MODE=send_recv

# Data path weak scaling
DATA_PATH=/lustre/orion/csc613/proj-shared/balin/Nek/GNN/weak_scale_data/500k_frontier/${PROCS}/gnn_outputs_poly_5/

EXE=/lustre/orion/csc613/proj-shared/balin/Nek/GNN/GNN/NekRS-ML/main.py
ARGS="epochs=120 backend=nccl halo_swap_mode=${HALO_SWAP_MODE} gnn_outputs_path=${DATA_PATH} master_port=3442 master_addr=${MASTER_ADDR}"
echo Running script $EXE
echo with arguments $ARGS
echo
echo `date`
srun -N$NODES -n$PROCS --ntasks-per-node=$PROCS_PER_NODE -c7 --gpu-bind=closest \
    python $EXE $ARGS
echo `date`



