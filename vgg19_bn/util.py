import torch

def to_f16(t: torch.Tensor):
    return t.type(dtype=torch.float16)

def to_f32(t: torch.Tensor):
    return t.type(dtype=torch.float32)
