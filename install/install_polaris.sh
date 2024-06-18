#!/bin/bash

# New conda env
module use /soft/modulefiles
module load conda/2024-04-29
conda activate
conda create -y -p /eagle/datascience/balin/SimAI-Bench/conda/clone --solver=libmamba --clone base
conda activate /eagle/datascience/balin/SimAI-Bench/conda/clone

# Install ML packages
pip uninstall -y torch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install other packages
#pip install mpipartition

