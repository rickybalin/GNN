import os
import sys
import socket
from typing import Optional, Union, Callable
import numpy as np
import time

try:
    #import mpi4py
    #mpi4py.rc.initialize = False
    from mpi4py import MPI
    WITH_DDP = True
except ModuleNotFoundError as e:
    WITH_DDP = False
    pass

import torch
try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError as e:
    pass
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.nn as distnn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import oneccl_bindings_for_pytorch as ccl
except ModuleNotFoundError as e:
    pass

TORCH_FLOAT_DTYPE = torch.float32

# Get MPI:
if WITH_DDP:
    LOCAL_RANK = int(os.getenv("PALS_LOCAL_RANKID"))
    #LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    COMM = MPI.COMM_WORLD

    try:
        WITH_CUDA = torch.cuda.is_available()
        if RANK == 0: print('Running on CUDA devices',flush=True)
    except:
        WITH_CUDA = False
        pass

    try:
        WITH_XPU = torch.xpu.is_available()
        if RANK == 0: print('Running on XPU devices',flush=True)
    except:
        WITH_XPU = False
        pass

    if WITH_CUDA:
        DEVICE = torch.device('cuda')
        N_DEVICES = torch.cuda.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES>1 else 0
        torch.cuda.set_device(DEVICE_ID)
    elif WITH_XPU:
        DEVICE = torch.device('xpu')
        N_DEVICES = torch.xpu.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES>1 else 0
        torch.xpu.set_device(DEVICE_ID)
    else:
        DEVICE = torch.device('cpu')
        DEVICE_ID = 'cpu'
        if RANK == 0: print('Running on CPU devices',flush=True)

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    # -----------------------------------------------------------
    # NOTE: Get the hostname of the master node, and broadcast
    # it to all other nodes It will want the master address too,
    # which we'll broadcast:
    # -----------------------------------------------------------
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)
else:
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    print('MPI Initialization failed!',flush=True)


def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)
    elif WITH_XPU:
        backend = 'ccl' if backend is None else str(backend)
    else:
        backend = 'gloo' if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )


def cleanup() -> None:
    dist.destroy_process_group()


def get_neighbors(args) -> List[int]:
    neighbors = []
    if args.all_to_all_buff == 'optimized':
        rank_list = [i for i in range(SIZE)]
        while len(neighbors)<args.num_neighbors:
            rank = random.choice(rank_list)
            if rank not in neighbors and rank!=RANK:
                neighbors.append(rank)
    return neighbors


def build_buffers(args, neighbors):
    buff_send = [torch.empty(0, device=DEVICE)] * SIZE
    buff_recv = [torch.empty(0, device=DEVICE)] * SIZE
    buff_send_sz = [0] * SIZE
    buff_recv_sz = [0] * SIZE

    if args.all_to_all_buff == 'naive':
        for i in range(SIZE):
            buff_send[i] = torch.empty([args.num_nodes, args.n_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_send_sz[i] = torch.numel(buff_send[i])*buff_send[i].element_size()
            buff_recv[i] = torch.empty([args.num_nodes, args.n_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_recv_sz[i] = torch.numel(buff_recv[i])*buff_recv[i].element_size()
    elif args.all_to_all_buff == 'optimized':
        for i in neighboring_procs:
            buff_send[i] = torch.empty([int(n_nodes_to_exchange[i]), n_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_send_sz[i] = torch.numel(buff_send[i])*buff_send[i].element_size()
            buff_recv[i] = torch.empty([int(n_nodes_to_exchange[i]), n_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_recv_sz[i] = torch.numel(buff_recv[i])*buff_recv[i].element_size()

    # Print information about the buffers
    print('[RANK %d]: Created send and receive buffers for %s halo exchange:' %(RANK,args.all_to_all_buff))
    print(f'[RANK {RANK}]: Send buffers of size {buff_send_sz}')
    print(f'[RANK {RANK}]: Receive buffers of size {buff_recv_sz}')

    return [buff_send, buff_recv]


def halo_test(args, neighbors, buffers) -> None:
    buff_send_safe = buffers[0]
    buff_recv_safe = buffers[1]
    buff_send = buff_send_safe
    buff_recv = buff_recv_safe

    times = []
    for itr in range(args.iterations):
        # initialize the buffers 
        for i in range(SIZE):
            buff_send[i] = torch.empty_like(buff_send_safe[i])
            buff_recv[i] = torch.empty_like(buffer_recv_safe[i])
        
        # fill in the non-empty buffers
        if args.all_to_all_buff == 'naive':
            for i in range(SIZE):
                buff_send[i] = torch.randn_like(buff_send[i])
        elif args.all_to_all_buff == 'optimized':
            for i in neighboring_procs:
                buff_send[i] = torch.randn_like(buff_send[i])

        # Perform the all_to_all
        tic = perf_counter()
        distnn.all_to_all(buff_recv, buff_send)
        toc = perf_counter()
        times.append(toc-tic)

    

def main() -> None:
    # Say hi
    print(f'Hello from rank {RANK}/{SIZE}, local rank {LOCAL_RANK}, on device {DEVICE}:{DEVICE_ID} out of {N_DEVICES}', flush=True)

    # Parse arguments
    parser = ArgumentParser(description='GNN for ML Surrogate Modeling for CFD')
    parser.add_argument('--all_to_all_buff', default='naive', type=str, choices=['naive','optimized'], help='Type of all_to_all buffers')
    parser.add_argument('--num_nodes', default=1638400, type=int, help='Number of input nodes (rows) to the all_to_all')
    parser.add_argument('--num_features', default=5, type=int, help='Number of input features (columns) to the all_to_all')
    parser.add_argument('--num_neighbors', default=0, type=int, help='Number of neighbors involved in the all_to_all')
    args = parser.parse_args()
    assert args.num_neighbors<=SIZE, 'Number of neighbors must be less than or equal to the number of ranks'
    if RANK == 0:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('RUNNING WITH INPUTS:')
        print(args)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if WITH_DDP:
        init_process_group(RANK, SIZE)
        
        neighbors = get_neighbors(args)
        buffers = build_buffers(args, neighbors)
        halo_test(args, neighbors, buffers)
        
        cleanup()


if __name__ == '__main__':
    main()
