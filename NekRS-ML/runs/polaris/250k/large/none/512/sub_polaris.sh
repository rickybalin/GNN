#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N gnn_scale
#PBS -l walltime=00:20:00
#PBS -l select=128:ncpus=64:ngpus=4
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -A datascience
##PBS -q prod
#PBS -q R2036270
##PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -V
##PBS -m be
##PBS -M rbalin@anl.gov

cd $PBS_O_WORKDIR
module use /soft/modulefiles
module load conda/2024-04-29
#conda activate /eagle/datascience/balin/SimAI-Bench/conda/clone
conda activate /lus/eagle/projects/datascience/balin/Nek/GNN/env/gnn
source /lus/eagle/projects/datascience/balin/Nek/GNN/env/_pyg/bin/activate

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

# for all_to_all
#export FI_CXI_RX_MATCH_MODE=software
#export FI_CXI_RDZV_PROTO=alt_read
#export FI_CXI_REQ_BUF_SIZE=8388608

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=4
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

# Halo swap mode
HALO_SWAP_MODE=none
#HALO_SWAP_MODE=all_to_all
#HALO_SWAP_MODE=all_to_all_opt
#HALO_SWAP_MODE=send_recv

# Data path strong scaling
#DATA_PATH=/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/gnn_outputs_distributed_gnn/gnn_outputs_poly_3/

# Data path weak scaling
#DATA_PATH=/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv_weak_scaling/ne_16_v2/gnn_outputs_poly_5/
DATA_PATH=/eagle/datascience/balin/Nek/GNN/weak_scale_data/250k_polaris/${PROCS}/gnn_outputs_poly_5/

EXE=/eagle/datascience/balin/Nek/GNN/GNN/NekRS-ML/main.py
ARGS="backend=nccl halo_swap_mode=${HALO_SWAP_MODE} gnn_outputs_path=${DATA_PATH}"
echo Running script $EXE
echo with arguments $ARGS
echo
echo `date`
mpiexec --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python $EXE ${ARGS} 
echo `date`



