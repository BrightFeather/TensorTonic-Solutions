import torch
def discounted_returns(rewards, gamma):
    """
    Returns: list of G_t values, one per timestep, each rounded to 4 decimals
    """
    r = torch.as_tensor(rewards, dtype=torch.float32)
    G = torch.empty_like(r)
    running = torch.zeros(r.shape[:-1], dtype=r.dtype)
    for t in range(r.shape[-1] - 1, -1, -1):
        running = r[..., t] + gamma * running
        G[..., t] = running
    return G
    



