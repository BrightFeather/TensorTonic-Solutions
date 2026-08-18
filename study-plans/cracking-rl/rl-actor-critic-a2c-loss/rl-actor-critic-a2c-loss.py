import torch
def a2c_loss(log_probs, advantages, values, returns, entropies, value_coef=0.5, entropy_coef=0.01):
    """
    Returns: float, total A2C loss rounded to 4 decimals
    """
    log_probs = torch.tensor(log_probs, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    entropies = torch.tensor(entropies, dtype=torch.float32)
    T = advantages.shape[-1]
    L_policy = -1/T * log_probs @ advantages
    L_v = 1/T * torch.sum(torch.square(returns - values))
    return L_policy + value_coef * L_v - entropy_coef * torch.mean(entropies)
