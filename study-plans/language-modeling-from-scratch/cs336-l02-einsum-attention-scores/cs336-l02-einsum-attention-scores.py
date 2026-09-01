import torch
import math

def attention_scores(q, k, num_heads):
    """
    Returns: tensor of shape (batch, heads, query_length, key_length)
    """
    batch, query_length, d_model = q.shape
    key_length = k.shape[1]
    d_h = d_model // num_heads
    
    # q = q.reshape(batch, query_length, num_heads, d_h).transpose(1,2)
    # k = k.reshape(batch, key_length, num_heads, d_h).transpose(1,2)
    # return q @ k.transpose(-2,-1) / math.sqrt(d_h)
    
    q = q.reshape(batch, query_length, num_heads, d_h)
    k = k.reshape(batch, key_length, num_heads, d_h)
    return torch.einsum('bqhd,bkhd->bhqk',q,k) / math.sqrt(d_h)
