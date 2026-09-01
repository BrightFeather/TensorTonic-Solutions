import torch
import torch.nn.functional as F

def parameter_matched_swiglu(x, w_g, w_v, w_o, base_params):
    """
    Returns: dictionary containing output, hidden_width, and parameter_count
    """
    model_width = x.shape[-1]
    h_max = w_g.shape[-1]
    h = min(max(math.floor(base_params / (3 * model_width) + 0.5),1),h_max)
    w_g, w_v, w_o = w_g[...,:h], w_v[...,:h], w_o[:h,...]
    y = (F.silu(x @ w_g) * (x @ w_v)) @ w_o
    return {
        "output": y, 
        "hidden_width":h, 
        "parameter_count":h*3*model_width
    }
    
