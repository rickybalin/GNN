#!/bin/bash

module use /soft/preview-modulefiles/24.086.0
module load frameworks/2024.04.15.002
source /gila/Aurora_deployment/balin/Nek/venv/_pyg/bin/activate

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=2
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
export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"

#mpiexec --pmi=pmix -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 ./affinity_aurora.sh python main.py --device=xpu --iterations=100 --problem_size=medium 
#mpiexec --pmi=pmix -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 ./affinity_aurora.sh python main.py --device=xpu --iterations=10 --problem_size=large
mpiexec --pmi=pmix --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=${CPU_BIND} python main.py --device=xpu --iterations=10 --problem_size=large 2>&1 | tee out.log



