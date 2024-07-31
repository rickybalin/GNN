#!/bin/bash

JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')

for i in 1 2 4 8
do
    echo Running training on $i ranks ...
    cd $i
    ./run_polaris.sh 2>&2 | tee gnn_sale.o$JOBID
    cd ..
    echo 
    echo Done
    echo 
    echo
done

