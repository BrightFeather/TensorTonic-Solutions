from math import prod
def flop_estimator(matmuls, attention_flops=0):
    """
    Returns: dictionary containing exact forward, backward, and total FLOP counts
    """
    F_foward = 2 * sum(prod(a) for a in matmuls) + attention_flops
    F_backward = 2*F_foward
    F_total = F_foward + F_backward
    return {"forward_flops":F_foward,"backward_flops":F_backward,"total_flops":F_total}
