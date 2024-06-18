import os
import socket
import datetime
from typing import Tuple
from argparse import ArgumentParser
import math
from time import perf_counter
import numpy as np

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpipartition import Partition

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

import gnn

PI = math.pi
MPI_OPS = {
        "sum": MPI.SUM,
        "min": MPI.MINLOC,
        "max": MPI.MAXLOC
}

def load_model(args, data) -> torch.nn.Module:
    """ Instantiate the GNN model
    
    :param args: config arguments 
    :type args: ...
    :return: instantiated GNN model
    :rtype: torch.nn.Module
    """
    if args.problem_size=='medium':
        hidden_channels = 16
        n_mlp_hidden_layers = 2
        n_messagePassing_layers = 6
    elif args.problem_size=='large':
        hidden_channels = 32
        n_mlp_hidden_layers = 5
        n_messagePassing_layers = 8

    model = gnn.MP_GNN(
            input_node_channels = data.x.shape[1],
            input_edge_channels = data.x.shape[1] + data.pos.shape[1] + 1,
            hidden_channels = hidden_channels,
            output_node_channels = data.y.shape[1],
            n_mlp_hidden_layers = n_mlp_hidden_layers,
            n_messagePassing_layers = n_messagePassing_layers,
            name = 'gnn')
  
    if (args.precision == "fp32" or args.precision == "tf32"):
        model.float()

    return model

def generate_data(args, comm) -> Data:
    """ Generate synthetic training data for the GNN model
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    random_seed = 12345 + 1000*rank
    if (args.problem_size=="medium"):
        N = 256
        n_samples = N**2
        ndIn = 2
        ndTot = 4
        partition = Partition(dimensions=2, comm=comm)
        part_origin = partition.origin
        part_extent = partition.extent
        x = np.linspace(part_origin[0],part_origin[0]+part_extent[0],num=N)*4*PI-2*PI
        y = np.linspace(part_origin[1],part_origin[1]+part_extent[1],num=N)*4*PI-2*PI
        x, y = np.meshgrid(x, y)
        coords = np.vstack((x.flatten(),y.flatten())).T
        r = np.sqrt(x**2+y**2)
        u = np.sin(2.0*r-0)/(r+1.0)
        udt = np.sin(2.0*r-0.01)/(r+1.0)
        v = np.cos(2.0*r-0)/(r+1.0)
        vdt = np.cos(2.0*r-0.01)/(r+1.0)
        inputs = np.empty((n_samples,ndIn))
        outputs = np.empty((n_samples,ndTot-ndIn))
        inputs[:,0] = u.flatten()
        inputs[:,1] = v.flatten() 
        outputs[:,0] = udt.flatten() 
        outputs[:,1] = vdt.flatten()
    elif (args.problem_size=="large"):
        N = 100
        n_samples = N**3
        ndIn = 3
        ndTot = 6
        partition = Partition(dimensions=3, comm=comm)
        part_origin = partition.origin
        part_extent = partition.extent
        x = np.linspace(part_origin[0],part_origin[0]+part_extent[0],num=N)*4*PI-2*PI
        y = np.linspace(part_origin[1],part_origin[1]+part_extent[1],num=N)*4*PI-2*PI
        z = np.linspace(part_origin[2],part_origin[2]+part_extent[2],num=N)*4*PI-2*PI
        x, y, z = np.meshgrid(x, y, z)
        coords = np.vstack((x.flatten(),y.flatten(),z.flatten())).T
        r = np.sqrt(x**2+y**2+z**2)
        u = np.sin(2.0*r-0)/(r+1.0)
        udt = np.sin(2.0*r-0.01)/(r+1.0)
        v = np.cos(2.0*r-0)/(r+1.0)
        vdt = np.cos(2.0*r-0.01)/(r+1.0)
        w = np.sin(2.0*r-0)**2/(r+1.0)
        wdt = np.sin(2.0*r-0.01)**2/(r+1.0)
        inputs = np.empty((n_samples,ndIn))
        outputs = np.empty((n_samples,ndTot-ndIn))
        inputs[:,0] = u.flatten()
        inputs[:,1] = v.flatten()
        inputs[:,2] = w.flatten()
        outputs[:,0] = udt.flatten()
        outputs[:,1] = vdt.flatten()
        outputs[:,2] = wdt.flatten()
        
    if (args.precision == "fp32" or args.precision == "tf32"):
        dtype = torch.float32

    edge_index = knn_graph(torch.from_numpy(coords), k=2, loop=False)
    data = Data(x=torch.from_numpy(inputs).type(dtype), y=torch.from_numpy(outputs).type(dtype), 
                pos=torch.from_numpy(coords).type(dtype), edge_index=edge_index)
    return data

def count_weights(model) -> int:
    """ Count the number of trainable parameters in the model
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params

def train(args, model, data, comm) -> dict:
    """ Train the GNN model
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    times = {
        'training_loop': 0.0,
        'train_tot': 0.0,
        'train_iter': [],
        'throughput_iter': [],
    }

    # Set up optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate*size)
    loss_fn = torch.nn.MSELoss()

    # Loop over iterations
    if rank==0:
        print('\nStarting training loop ... ', flush=True)
    num_graph_nodes = data.x.shape[0]
    model.train()
    tic_l = perf_counter()
    for iteration in range(args.iterations):
        tic_t = perf_counter()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.pos)
        loss = loss_fn(pred, data.y)
        loss.backward()
        optimizer.step()
        toc_t = perf_counter()

        if args.device=='xpu':
            dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
            if rank==0: loss /= size
        else:
            dist.reduce(loss, 0, op=dist.ReduceOp.AVG)
        if rank==0:
            print(f'[{iteration}]: avg_loss = {loss:>4e}', flush=True)

        if iteration>0:
            t_train = toc_t - tic_t
            times['train_tot'] += t_train
            times['train_iter'].append(t_train)
            times['throughput_iter'].append(num_graph_nodes/t_train)

    toc_l = perf_counter()
    times['training_loop'] = toc_l - tic_l 

    return times

def print_fom(comm, times):
    """ Calculate and print performance FOM
    """
    time_stats = {}
    for key, val in times.items():
        if type(val)==list:
            collected_arr = np.zeros((len(val)*comm.Get_size()))
            comm.Gather(np.array(val),collected_arr,root=0)
            avg = np.mean(collected_arr)
            std = np.std(collected_arr)
            minn = np.amin(collected_arr); min_loc = [minn, 0]
            maxx = np.amax(collected_arr); max_loc = [maxx, 0]
            summ = np.sum(collected_arr)
        else:
            summ = comm.allreduce(np.array(val), op=MPI_OPS["sum"])
            avg = summ / comm.Get_size()
            tmp = np.power(np.array(val - avg),2)
            std = comm.allreduce(tmp, op=MPI_OPS["sum"])
            std = std / comm.Get_size()
            std = np.sqrt(std)
            min_loc = comm.allreduce((val,comm.Get_rank()), op=MPI_OPS["min"])
            max_loc = comm.allreduce((val,comm.Get_rank()), op=MPI_OPS["max"])
        stats = {
                "avg": avg,
                "std": std,
                "sum": summ,
                "min": [min_loc[0],min_loc[1]],
                "max": [max_loc[0],max_loc[1]]
        }
        time_stats[key] = stats

    # Average parallel training throughout
    sum_across_ranks = np.zeros((len(times['throughput_iter'])))
    comm.Reduce(np.array(times['throughput_iter']),sum_across_ranks,op=MPI_OPS["sum"])
    avg_par_throughput = np.mean(sum_across_ranks)

    if comm.rank==0:
        print('\nSummary of performance data:')
        for key, val in time_stats.items():
            stats_string = f": min = {val['min'][0]:>6e} , " + \
                           f"max = {val['max'][0]:>6e} , " + \
                           f"avg = {val['avg']:>6e} , " + \
                           f"std = {val['std']:>6e} "
            print(f"{key} [s] " + stats_string, flush=True) 
        print(f"Average parallel training throughout [nodes/s] : {avg_par_throughput:>6e}")


if __name__ == '__main__':
    # MPI Init
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    rankl = int(os.getenv("PALS_LOCAL_RANKID"))
    comm.Barrier()
    print(f'Hello from rank {rank}/{size}, local rank {rankl} on {name}', flush=True)

    # Parse arguments
    parser = ArgumentParser(description='GNN for ML Surrogate Modeling for CFD')
    parser.add_argument('--device', default="cuda", type=str, help='Device to use for training (cpu,cuda,xpu)')
    parser.add_argument('--problem_size', default="medium", type=str, help='Size of ML problem to train (medium)')
    parser.add_argument('--iterations', default=10, type=int, help='Number of optimizer iterations to perform')
    parser.add_argument('--precision', default="fp32", type=str, help='Floating point precision')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
    args = parser.parse_args()

    # Intel imports
    if args.device=='xpu':
        try:
            import intel_extension_for_pytorch as ipex
        except ModuleNotFoundError as e:
            if rank==0: print(e, flush=True)
        try:
            import oneccl_bindings_for_pytorch as ccl
        except ModuleNotFoundError as e:
            if rank==0: print(e, flush=True)

    # Initialize Torch Distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
    master_addr = socket.gethostname() if rank == 0 else None
    master_addr = comm.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(2345)
    if (args.device=='cpu'): backend = 'gloo'
    elif (args.device=='cuda'): backend = 'nccl'
    elif (args.device=='xpu'): backend = 'ccl'
    dist.init_process_group(backend,
                            rank=int(rank),
                            world_size=int(size),
                            init_method='env://',
                            timeout=datetime.timedelta(seconds=120))

    # Load model and data
    data = generate_data(args, comm)
    model = load_model(args, data)
    n_params = count_weights(model)
    if (rank == 0):
        print(f"\nLoaded data and model with {n_params} parameters\n", flush=True)

    # Set device to run on and offload model
    device = torch.device(args.device)
    torch.set_num_threads(1)
    if (args.device == 'cuda'):
        if torch.cuda.is_available():
            cuda_id = rankl if torch.cuda.device_count()>1 else 0
            assert cuda_id>=0 and cuda_id<torch.cuda.device_count(), \
                   f"Assert failed: cuda_id={cuda_id} and {torch.cuda.device_count()} available devices"
            torch.cuda.set_device(cuda_id)
        else:
            print(f"[{rank}]: no cuda devices available, cuda.device_count={torch.cuda.device_count()}", flush=True)
    elif (args.device=='xpu'):
        if torch.xpu.is_available():
            xpu_id = rankl if torch.xpu.device_count()>1 else 0
            assert xpu_id>=0 and xpu_id<torch.xpu.device_count(), \
                   f"Assert failed: xpu_id={xpu_id} and {torch.xpu.device_count()} available devices"
            torch.xpu.set_device(xpu_id)
        else:
            print(f"[{comm.rank}]: no XPU devices available, xpu.device_count={torch.xpu.device_count()}", flush=True)
    if (args.device != 'cpu'):
        model.to(device)
        data = data.to(device)
    if (rank == 0):
        print(f"Running on device: {args.device} \n", flush=True)

    # Wrap model with DDP
    model = DDP(model) 

    # Perform training
    times = train(args, model, data, comm)

    # Calculate and print FOM
    if args.iterations>1:
        print_fom(comm, times)

    # Finalize
    dist.destroy_process_group()
    MPI.Finalize()
