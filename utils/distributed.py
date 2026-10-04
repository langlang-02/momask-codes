import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def dist_is_initialized():
    return dist.is_available() and dist.is_initialized()


def init_distributed_mode(opt):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(opt, "local_rank", 0)))

    opt.distributed = world_size > 1
    opt.world_size = world_size
    opt.rank = rank
    opt.local_rank = local_rank
    opt.is_main_process = rank == 0

    if opt.distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            opt.gpu_id = local_rank
            opt.device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            opt.gpu_id = -1
            opt.device = torch.device("cpu")
            backend = "gloo"

        if not dist_is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        opt.world_size = dist.get_world_size()
        opt.rank = dist.get_rank()
        opt.is_main_process = opt.rank == 0
    else:
        if getattr(opt, "gpu_id", -1) != -1 and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_id)
            opt.device = torch.device("cuda", opt.gpu_id)
        else:
            opt.gpu_id = -1
            opt.device = torch.device("cpu")

    return opt


def cleanup_distributed():
    if dist_is_initialized():
        dist.destroy_process_group()


def is_main_process(opt=None):
    if opt is not None:
        return getattr(opt, "is_main_process", True)
    return not dist_is_initialized() or dist.get_rank() == 0


def barrier():
    if dist_is_initialized():
        dist.barrier()


def wrap_ddp(model, opt, find_unused_parameters=False, broadcast_buffers=True):
    if not getattr(opt, "distributed", False):
        return model

    if opt.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
        )
    return DistributedDataParallel(
        model,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_state_dict_compat(model, state_dict, strict=True):
    target = unwrap_model(model)
    try:
        return target.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if all(key.startswith("module.") for key in state_dict):
            stripped = {key[len("module."):]: value for key, value in state_dict.items()}
            return target.load_state_dict(stripped, strict=strict)
        raise


def set_loader_epoch(loader, epoch):
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def reduce_sum(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist_is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.detach().cpu().tolist()
