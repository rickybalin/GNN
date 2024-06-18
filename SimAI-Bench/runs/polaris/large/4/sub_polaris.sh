#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N gnn_scale
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -A datascience
##PBS -q prod
##PBS -q preemptable
##PBS -q debug-scaling
#PBS -q debug
#PBS -V
##PBS -m be
##PBS -M rbalin@anl.gov

cd $PBS_O_WORKDIR
module use /soft/modulefiles
module load conda/2024-04-29
conda activate /eagle/datascience/balin/SimAI-Bench/conda/clone
echo Loaded modules:
module list
echo
echo Torch versions
pip list | grep torch
echo

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=4
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

EXE=/eagle/datascience/balin/Nek/GNN/GNN/SimAI-Bench/main.py
ARGS="--device=cuda --iterations=50 --problem_size=large"
echo Running script $EXE
echo with arguments $ARGS
mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python $EXE ${ARGS}



