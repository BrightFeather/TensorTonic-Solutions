import math
import torch
from torch.nn.functional import logsigmoid

def dpo_loss(log_pi_new_w, log_pi_new_l, log_pi_ref_w, log_pi_ref_l, beta):
    """
    Returns: float, DPO loss rounded to 4 decimals
    """
    log_pi_new_w = torch.as_tensor(log_pi_new_w, dtype=torch.float32)
    log_pi_new_l = torch.as_tensor(log_pi_new_l, dtype=torch.float32)
    log_pi_ref_w = torch.as_tensor(log_pi_ref_w, dtype=torch.float32)
    log_pi_ref_l = torch.as_tensor(log_pi_ref_l, dtype=torch.float32)
    return - torch.mean(logsigmoid(beta * ((log_pi_new_w - log_pi_ref_w) - (log_pi_new_l - log_pi_ref_l))))