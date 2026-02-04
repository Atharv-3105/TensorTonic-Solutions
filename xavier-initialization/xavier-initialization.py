def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    n = len(W)
    m = len(W[0])
    W_dash = [[0 for _ in range(m)]for _ in range(n)]

    #Step1: Xavier Uniform Bound=> Limit = Sqrt(6 / [f_in + f_out])
    deno = fan_in + fan_out
    Limit = (6 / deno) ** 0.5

    #Step2: Update Weights
    for i in range(n):
        for j in range(m):
            W_dash[i][j] = (W[i][j] * 2 * Limit) - Limit
    
    return W_dash