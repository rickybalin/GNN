#!/bin/bash

module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/5.7.1
module load craype-accel-amd-gfx90a

#source activate base
conda create -y -p $PWD/gnn python=3.10
source activate $PWD/gnn

#python3 -m venv --clear $PWD/_pyg --system-site-packages
#source $PWD/_pyg/bin/activate

pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7
pip install torch_geometric
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.2.2+cpu.html

MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

pip install mpipartition
pip install hydra-core
pip install matplotlib

