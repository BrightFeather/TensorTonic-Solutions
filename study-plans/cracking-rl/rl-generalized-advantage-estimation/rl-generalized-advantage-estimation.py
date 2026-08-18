import torch
def gae_advantages(rewards, values, gamma, lam, last_value=0.0):
    """
    Returns: list of T advantages rounded to 4 decimals
    """
    T = len(rewards)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    adv = torch.zeros_like(values, dtype=torch.float32)
    last_value = torch.as_tensor(last_value, dtype=torch.float32)
    V_next = torch.cat([values[...,1:], last_value.unsqueeze(-1)], -1)
    delta = rewards + gamma * V_next - values
    running = torch.zeros(rewards.shape[:-1], dtype=torch.float32)
    for i in range(values.shape[-1] - 1, -1, -1):
        running = delta[..., i] + gamma * lam * running
        adv[..., i] = running
    return adv
