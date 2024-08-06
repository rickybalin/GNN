#!/bin/bash 

module use /soft/modulefiles
module load conda/2024-04-29
#conda activate /eagle/datascience/balin/SimAI-Bench/conda/clone
conda activate /lus/eagle/projects/datascience/balin/Nek/GNN/env/gnn
source /lus/eagle/projects/datascience/balin/Nek/GNN/env/_pyg/bin/activate

echo Loaded modules:
module list
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

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=4
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

EXE=/eagle/datascience/balin/Nek/GNN/GNN/SimAI-Bench/main.py
ARGS="--device=cuda --iterations=5 --problem_size=large"
echo Running script $EXE
echo with arguments $ARGS
echo
echo `date`
mpiexec --envall --env TMPDIR=$PWD/tmpdir/ -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas -o nsys_report --stats=true --show-output=true python $EXE $ARGS
echo `date`



