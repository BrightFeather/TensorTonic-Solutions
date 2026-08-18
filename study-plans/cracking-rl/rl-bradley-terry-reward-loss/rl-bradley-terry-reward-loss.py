import math
import torch
from torch.nn.functional import logsigmoid

def bradley_terry_loss(r_chosen, r_rejected):
    """
    Returns: float, Bradley-Terry preference loss rounded to 4 decimals
    """
    r_chosen = torch.tensor(r_chosen, dtype=torch.float32)
    r_rejected = torch.tensor(r_rejected, dtype=torch.float32)
    return -torch.mean(logsigmoid(r_chosen - r_rejected))
