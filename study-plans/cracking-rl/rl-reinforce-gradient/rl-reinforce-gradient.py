import torch
def reinforce_loss(log_probs, returns):
    """
    Returns: float, REINFORCE policy loss rounded to 4 decimals
    """
    log_probs = torch.as_tensor(log_probs, dtype=torch.float32)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    return -1./returns.shape[-1] * (log_probs @ returns)