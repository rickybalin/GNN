#!/bin/bash

module use /soft/preview-modulefiles/24.086.0
module load frameworks/2024.04.15.002
source /gila/Aurora_deployment/balin/Nek/venv/_pyg/bin/activate

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=8
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

DATA_PATH=/gila/Aurora_deployment/balin/Nek/nekrs_cases/examples_v23_gnn/tgv/gnn_outputs_distributed_gnn/gnn_outputs_poly_1/

# OneCCL
#mpiexec --pmi=pmix -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 ./affinity_aurora.sh python main.py device=xpu backend=ccl halo_swap_mode=all_to_all gnn_outputs_path=${DATA_PATH} 2>&1 | tee out.log 
#mpiexec --pmi=pmix -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 ./affinity_aurora.sh python main.py device=xpu backend=ccl halo_swap_mode=none gnn_outputs_path=${DATA_PATH} 2>&1 | tee out.log 

# GLOO
mpiexec --pmi=pmix -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 ./affinity_aurora.sh python main.py device=xpu backend=gloo halo_swap_mode=none gnn_outputs_path=${DATA_PATH} 2>&1 | tee out.log 



