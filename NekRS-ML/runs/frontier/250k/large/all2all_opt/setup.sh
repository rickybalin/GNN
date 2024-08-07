#!/bin/bash

for NGPU in 4 8 16 32 64 128 256 512 1024
do
    if [[ -d $NGPU ]]; then
        cd $NGPU
    else
        mkdir $NGPU
        cd $NGPU
    fi

    cp ../sub_frontier.sl .
    NUM_NODES=$(( NGPU/4 ))
    sed -i "s/NUM_NODES/${NUM_NODES}/g" sub_frontier.sl

    cd .. 
done
