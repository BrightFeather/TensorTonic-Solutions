from math import prod

def memory_accountant(param_shapes, param_bytes_per_element, grad_bytes_per_element,
                       activation_shapes, activation_bytes_per_element,
                       optimizer, optimizer_bytes_per_element):
    """
    Returns: dictionary containing exact parameter, gradient, activation, optimizer, and total bytes
    """
    num_params = sum(prod(x) for x in param_shapes)
    num_activations = sum(prod(x) for x in activation_shapes)
    optimizer_bytes_per_param = {"sgd":0, "adagrad":1, "adam":2}
    
    return {
        'parameters': param_bytes_per_element * num_params, 
        'gradients': grad_bytes_per_element * num_params, 
        'activations': activation_bytes_per_element * num_activations,
        'optimizer_state': optimizer_bytes_per_element * num_params * optimizer_bytes_per_param[optimizer],
        'total': param_bytes_per_element * num_params + grad_bytes_per_element * num_params + activation_bytes_per_element * num_activations + optimizer_bytes_per_element * num_params * optimizer_bytes_per_param[optimizer]
    }
