import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """
    x_float = x.float()
    g_float = g.float()
    normalized = x * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True)+epsilon)
    return (normalized * g).to(x.dtype)
