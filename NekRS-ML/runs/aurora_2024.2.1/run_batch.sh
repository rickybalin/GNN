#!/bin/bash


JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')

for DIR in "all2all" "all2all_opt"
do
  for N in 24 48 96
  do
    cd ${DIR}/${N}
    ./run_aurora.sh 2>&1 | tee gnn_scale.o${JOBID}
    cd ../..
  done
done

