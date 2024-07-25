#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N all2all
#PBS -l walltime=00:30:00
#PBS -l select=16
#PBS -k doe
#PBS -j oe
#PBS -A Aurora_deployment
#PBS -q lustre_scaling
#PBS -V
##PBS -m be
##PBS -M rbalin@anl.gov 

cd $PBS_O_WORKDIR
module load frameworks/2024.1
source /lus/flare/projects/Aurora_deployment/balin/Nek/GNN/env/_pyg/bin/activate
echo Loaded modules:
module list
echo
echo Torch version: `python -c "import torch; print(torch.__version__)"`
echo IPEX version: `python -c "import torch;import intel_extension_for_pytorch as ipex;print(ipex.__version__)"`
echo Torch Geometric version: `python -c "import torch;import torch_geometric;print(torch_geometric.__version__)"`
echo

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

#export MPIR_CVAR_ENABLE_GPU=0
export CCL_KVS_GET_TIMEOUT=600
export CCL_PROCESS_LAUNCHER=pmix
export FI_CXI_OFLOW_BUF_SIZE=20971520 # increased 2MB to 20MB

export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi

export CCL_CONFIGURATION_PATH=""
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ROOT="/flare/Aurora_deployment/intel/ccl/ccl_3504f9b6_install"
export LD_LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/ccl_3504f9b6_install/lib:$LD_LIBRARY_PATH
export CPATH=/flare/Aurora_deployment/intel/ccl/ccl_3504f9b6_install/include:$CPATH
export LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/ccl_3504f9b6_install/lib:$LIBRARY_PATH

export CPU_BIND="list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96"
if [[ $PROCS_PER_NODE -eq 1 ]]; then
    echo Not setting oneCCL bindings
elif [[ $PROCS_PER_NODE -eq 2 ]]; then
    echo Setting oneCCL bindings for 2 ranks per node
    export CCL_WORKER_AFFINITY="5,13"
elif [[ $PROCS_PER_NODE -eq 6 ]]; then
    echo Setting oneCCL bindings for 6 ranks per node
    export CCL_WORKER_AFFINITY="5,13,21,29,37,45"
elif [[ $PROCS_PER_NODE -eq 12 ]]; then
    echo Setting oneCCL bindings for 12 ranks per node
    export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"
fi
echo

EXE=/flare/Aurora_deployment/balin/Nek/GNN/GNN/NekRS-ML/all2all.py
ARGS="--all_to_all_buff=optimized --num_neighbors=2 --iterations=50"
echo Running script $EXE
echo with arguments $ARGS
echo
echo `date`
mpiexec --pmi=pmix --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=${CPU_BIND} python $EXE ${ARGS}
echo `date`



