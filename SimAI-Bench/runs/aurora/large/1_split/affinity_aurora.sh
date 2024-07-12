#!/bin/bash

num_gpu=6
num_tile=2

# Get the RankID from different launcher
if [[ -v MPI_LOCALRANKID ]]; then
  _MPI_LRANKID=$MPI_LOCALRANKID
elif [[ -v PALS_LOCAL_RANKID ]]; then
  _MPI_LRANKID=$PALS_LOCAL_RANKID
  _MPI_RANKID=$PALS_RANKID
else
  display_help
fi

if [[ $_MPI_LRANKID -eq 0 ]]; then
  gpu_name=0.0
  export ZE_AFFINITY_MASK=0.0
elif [[ $_MPI_LRANKID -eq 1 ]]; then
  gpu_name=1.0
  export ZE_AFFINITY_MASK=1.0
fi

#gpu_id=$(((_MPI_LRANKID / num_tile) % num_gpu))
#tile_id=$((_MPI_LRANKID % num_tile))

unset EnableWalkerPartition
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
#export ZE_AFFINITY_MASK=$gpu_id.$tile_id
echo ?RANK= ${_MPI_RANKID} LOCAL_RANK= ${_MPI_LRANKID} gpu= ${gpu_name}?

exec "$@"
