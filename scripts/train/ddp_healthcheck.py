#!/usr/bin/env python3
"""Small single-node NCCL startup/all-reduce audit used by training launchers."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    value = torch.tensor(float(dist.get_rank() + 1), device="cuda")
    for _ in range(10):
        dist.all_reduce(value)
        value.fill_(float(dist.get_rank() + 1))
    dist.barrier()
    print(f"rank={dist.get_rank()} device={local_rank} nccl_health=ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
