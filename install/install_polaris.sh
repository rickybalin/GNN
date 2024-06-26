#!/bin/bash

# New conda env
module use /soft/modulefiles
module load conda/2024-04-29
conda activate
conda create -y -p /lus/eagle/projects/datascience/balin/Nek/GNN/env/gnn --solver=libmamba --clone base
conda activate /lus/eagle/projects/datascience/balin/Nek/GNN/env/gnn

# Uninstall ML packages from conda env
#pip uninstall -y torch
pip uninstall -y torch-sparse torch_spline_conv

# Re-Install ML packages in a venv
python3 -m venv --clear /lus/eagle/projects/datascience/balin/Nek/GNN/env/_pyg --system-site-packages
source /lus/eagle/projects/datascience/balin/Nek/GNN/env/_pyg/bin/activate

#pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
#pip install torch_geometric torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
CC=cc pip install torch_cluster==1.6.3

# Install other packages
pip install mpipartition

