import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pe = np.empty((seq_length, d_model))
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    for pos in range(seq_length):
        pe1 = np.sin(pos*div)
        pe2 = np.cos(pos*div)
        pe[pos] = np.vstack((pe1, pe2)).flatten('F')
    return pe
        