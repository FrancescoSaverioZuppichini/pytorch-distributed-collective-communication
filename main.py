import os
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def do_reduce(rank: int, size: int):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    # sending all tensors to rank 0 and sum them
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
    # can be dist.ReduceOp.PRODUCT, dist.ReduceOp.MAX, dist.ReduceOp.MIN
    # only rank 0 will have four
    print(f"[{rank}] data = {tensor[0]}")

def do_all_reduce(rank: int, size: int):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    # can be dist.ReduceOp.PRODUCT, dist.ReduceOp.MAX, dist.ReduceOp.MIN
    # will output 4 for all ranks
    print(f"[{rank}] data = {tensor[0]}")


def do_scatter(rank: int, size: int):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    tensor = torch.empty(1)
    # sending all tensors from rank 0 to the others
    if rank == 0:
        tensor_list = [torch.tensor([i + 1], dtype=torch.float32) for i in range(size)]
        # tensor_list = [tensor(1), tensor(2), tensor(3), tensor(4)]
        dist.scatter(tensor, scatter_list=tensor_list, src=0, group=group)
    else:
        dist.scatter(tensor, scatter_list=[], src=0, group=group)
    # each rank will have a tensor with their rank number
    print(f"[{rank}] data = {tensor[0]}")


def do_gather(rank: int, size: int):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    tensor = torch.tensor([rank], dtype=torch.float32)
    # sending all tensors from rank 0 to the others
    if rank == 0:
        # create an empty list we will use to hold the gathered values
        tensor_list = [torch.empty(1) for i in range(size)]
        dist.gather(tensor, gather_list=tensor_list, dst=0, group=group)
    else:
        dist.gather(tensor, gather_list=[], dst=0, group=group)
    # only rank 0 will have the tensors from the other processed
    # [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
    if rank == 0:
        print(f"[{rank}] data = {tensor_list}")


def do_all_gather(rank: int, size: int):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    tensor = torch.tensor([rank], dtype=torch.float32)
    # create an empty list we will use to hold the gathered values
    tensor_list = [torch.empty(1) for i in range(size)]
    # sending all tensors to the others
    dist.all_gather(tensor_list, tensor, group=group)
    # all ranks will have [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
    print(f"[{rank}] data = {tensor_list}")


def do_broadcast(rank: int, size: int):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    if rank == 0:
        tensor = torch.tensor([rank], dtype=torch.float32)
    else:
        tensor = torch.empty(1)
        # sending all tensors to the others
    dist.broadcast(tensor, src=0, group=group)
    # all ranks will have tensor([0.]) from rank 0
    print(f"[{rank}] data = {tensor}")


def hello_world(rank: int, size: int):
    print(f"[{rank}] say hi!")


def init_process(rank: int, size: int, fn: Callable[[int, int], None], backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, do_broadcast))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()