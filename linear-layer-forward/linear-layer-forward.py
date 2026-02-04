def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    n = len(X) #Number of Samples
    d_in = len(X[0])    #Input Dimension
    d_out = len(W[0])   #Output Dimension
    #Initialize Y of size(n x d_out)
    Y = [[0 for _ in range(d_out)] for _ in range(n)] 
    for i in range(n):
        for j in range(d_out):
            for k in range(d_in):
                Y[i][j] += (X[i][k] * W[k][j]) 
            Y[i][j] += b[j]
    
    return Y
           