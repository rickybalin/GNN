#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N gnn_scale
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -k doe
#PBS -j oe
#PBS -A Aurora_deployment
#PBS -q workq
#PBS -V
##PBS -m be
##PBS -M rbalin@anl.gov

cd $PBS_O_WORKDIR
module use /soft/preview-modulefiles/24.086.0
module load frameworks/2024.04.15.002
source /gila/Aurora_deployment/balin/Nek/venv/_pyg/bin/activate
echo Loaded modules:
module list
echo
echo Torch versions
pip list | grep torch
echo

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=12
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

EXE=./main.py
GPU_AFFINITY=./affinity_aurora.sh
ARGS="--device=xpu --iterations=50 --problem_size=large"
echo Running script $EXE
echo with arguments $ARGS
#mpiexec --pmi=pmix -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 $GPU_AFFINITY python $EXE ${ARGS}
mpiexec --pmi=pmix -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 python $EXE ${ARGS}



