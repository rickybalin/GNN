#!/bin/bash

#for NGPU in 4 8 16 32 64 128 256 512
#for NGPU in 4 8 16 32
for NGPU in 64 128 256 512
do
    cd $NGPU
    sbatch --export=NONE sub_frontier.sl 
    cd .. 
done
