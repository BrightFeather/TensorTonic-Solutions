import numpy as np
import torch

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here

    batch_size, length, d = Q.shape
    d_k = d // num_heads
    Q = np.dot(Q, W_q).reshape(batch_size, length, num_heads, d_k)
    K = np.dot(K, W_k).reshape(batch_size, length, num_heads, d_k)
    V = np.dot(V, W_v).reshape(batch_size, length, num_heads, d_k)
    Q, K, V = torch.as_tensor(Q), torch.as_tensor(K), torch.as_tensor(V) 

    attention_score = torch.softmax(torch.einsum('blhk,bshk->bhls', Q, K)/ np.sqrt(d_k), axis=-1)
    attention = torch.einsum('bhls,bshk->blhk', attention_score, V)
    out = attention.reshape(batch_size, length, d) @ W_o
    return np.array(out)