#!/bin//bash 

module use /soft/modulefiles
module load jax/0.4.29-dev

python3 -m venv --clear /lus/eagle/projects/datascience/balin/Nek/GNN/env/_pyg_old --system-site-packages
source /lus/eagle/projects/datascience/balin/Nek/GNN/env/_pyg_old/bin/activate

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric==2.5.3
pip install --force-reinstall --no-cache-dir pyg_lib==0.4.0+pt112cu113 torch_scatter==2.1.0+pt112cu113 torch_sparse==0.6.16+pt112cu113 torch_cluster==1.6.0+pt112cu113 torch_spline_conv==1.2.1+pt112cu113 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

pip install matplotlib
 
