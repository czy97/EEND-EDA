import os
import torch.distributed as dist
import torch
import random


def getoneNode():
    nodelist = os.environ['SLURM_JOB_NODELIST']
    nodelist = nodelist.strip().split(',')[0]
    import re
    text = re.split('[-\[\]]', nodelist)
    if ('' in text):
        text.remove('')
    return text[0] + '-' + text[1] + '-' + text[2]


def dist_init(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    dist.init_process_group("nccl", init_method=host_addr_full,
                            rank=rank, world_size=world_size)
    assert dist.is_initialized()


def init_ddp(gpu_list='[0]'):
    if isinstance(gpu_list, int):
        gpu_list = [gpu_list]

    if 'WORLD_SIZE' in os.environ:
        # using torchrun to start:
        # egs: torchrun --standalone --rdzv_endpoint=localhost:$PORT_k --nnodes=1 --nproc_per_node=2 train.py --config conf/config.yaml --gpu_list '[0,1]'
        host_addr = 'localhost'
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_id = int(gpu_list[rank])
        # dist_init(host_addr, rank, local_rank,
                  # world_size, port)
        dist.init_process_group(backend='nccl')
    elif 'SLURM_LOCALID' in os.environ:
        # start process using slurm
        host_addr = getoneNode()
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        dist_init(host_addr, rank, local_rank,
                  world_size, '2' + os.environ['SLURM_JOBID'][-4:])
        gpu_id = local_rank
    else:
        # run locally with only one process
        host_addr = 'localhost'
        rank = 0
        local_rank = 0
        world_size = 1
        gpu_id = int(gpu_list[rank])
        dist_init(host_addr, rank, local_rank,
                  world_size, 8888 + random.randint(0, 1000))


    return gpu_id
