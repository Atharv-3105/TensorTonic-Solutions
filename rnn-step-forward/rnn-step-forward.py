import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    value1 = np.dot(x_t, Wx)
    value2 = np.dot(h_prev, Wh)
    value3 = value1 + value2 + b
    return np.tanh(value3)
