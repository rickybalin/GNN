#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N genbox
#PBS -l walltime=01:00:00
#PBS -l select=4:ncpus=64:ngpus=4
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -A datascience
##PBS -q prod
##PBS -q preemptable
#PBS -q debug-scaling
##PBS -q debug
#PBS -V
##PBS -m be
##PBS -M rbalin@anl.gov

JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')

for CASE in none all2all all2all_opt
do
    cd $CASE

    #for i in 1 2 4 8
    for i in 16
    do
        echo Running training with $CASE on $i ranks ...
        cd $i
        ./run_polaris.sh 2>&2 | tee gnn_sale.o$JOBID
        cd ..
        echo 
        echo Done
        echo 
        echo
    done

    cd ..
done

