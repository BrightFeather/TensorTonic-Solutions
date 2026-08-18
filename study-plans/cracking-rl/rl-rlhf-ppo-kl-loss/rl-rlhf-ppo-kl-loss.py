import torch
def rlhf_ppo_kl_loss(log_probs_new, log_probs_old, log_probs_ref, advantages, clip_eps, kl_coef):
    """
    Returns: float, RLHF PPO loss with KL penalty rounded to 4 decimals
    """
    log_probs_new = torch.tensor(log_probs_new, dtype=torch.float32)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
    log_probs_ref = torch.tensor(log_probs_ref, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    r = torch.exp(log_probs_new - log_probs_old)
    clipped = torch.clip(r, 1-clip_eps, 1+clip_eps)
    L_ppo = -torch.minimum(r * advantages, clipped * advantages).mean().item()
    KL = torch.mean(log_probs_new - log_probs_ref)
    return L_ppo + kl_coef * KL
    
 