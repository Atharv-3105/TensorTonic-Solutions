import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    denomi = 1 + np.exp(-x)
    result = 1 / denomi
    return result
    