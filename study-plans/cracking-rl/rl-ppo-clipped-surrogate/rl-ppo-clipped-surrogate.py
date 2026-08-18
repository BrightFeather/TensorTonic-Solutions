import math
import torch
def ppo_clipped_loss(log_probs_new, log_probs_old, advantages, clip_eps):
    """
    Returns: float, PPO clipped surrogate loss rounded to 4 decimals
    """
    log_probs_new = torch.tensor(log_probs_new, dtype=torch.float32)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    r = torch.exp(log_probs_new - log_probs_old)
    clipped = torch.clip(r, 1-clip_eps, 1+clip_eps)
    return -torch.minimum(r * advantages, clipped * advantages).mean().item()