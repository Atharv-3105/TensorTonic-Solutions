def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    n = len(W)
    m = len(W[0])
    W_dash = [[0 for _ in range(m)] for _ in range(n)]
    #Step1: Compute the He Uniform bound (L = [6/f_in]^0.5)
    Limit = (6 / fan_in) ** 0.5 


    #Change weights from range [0,1] to [-L,L]
    for i in range(n):
        for j in range(m):
            W_dash[i][j] = (W[i][j] * 2 * Limit) - Limit

    return W_dash