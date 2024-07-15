#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N gnn_scale
#PBS -l walltime=00:30:00
#PBS -l select=256:ncpus=64:ngpus=4
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -A datascience
#PBS -q prod
##PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -V
##PBS -m be
##PBS -M rbalin@anl.gov

cd $PBS_O_WORKDIR
module use /soft/modulefiles
module load conda/2024-04-29
##conda activate /eagle/datascience/balin/SimAI-Bench/conda/clone
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
#export NCCL_NET="AWS Libfabric"
#export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=4
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ranks per node: $PROCS_PER_NODE
echo Number of total ranks: $PROCS
echo

EXE=/lus/eagle/projects/datascience/balin/Nek/GNN/GNN/SimAI-Bench/main.py
ARGS="--device=cuda --iterations=100 --problem_size=large"
echo Running script $EXE
echo with arguments $ARGS
echo
echo `date`
mpiexec --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python $EXE ${ARGS}
echo `date`


