import torch
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask = None,
) -> torch.Tensor:
    """
    Returns a float32 attention tensor with shape (batch, query length, value width).
    """
    dk = query.shape[-1]
    scores = (query @ key.transpose(-1,-2)) / (dk ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)
