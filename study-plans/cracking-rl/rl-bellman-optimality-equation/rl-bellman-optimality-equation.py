import torch
def bellman_optimality_backup(P, R, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    P, R, V = torch.as_tensor(P, dtype=torch.float32), torch.as_tensor(R, dtype=torch.float32), torch.as_tensor(V, dtype=torch.float32)
    return (P * (R + gamma * V)).sum(-1).amax(1)
