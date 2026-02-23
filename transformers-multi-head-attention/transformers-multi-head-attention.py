import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    query = Q @ W_q
    key = K @ W_k
    value = V @ W_v

    #Split into multi-heads
    query = query.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)
    key = key.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)
    value = value.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)

    #Compute scaled dot-product attention
    key_transpose = key.transpose(0,1,3,2) #(batch_Size, num_heads,d_k, seq_len)
    scores = (query @ key_transpose) / (np.sqrt(d_k))
    attn = softmax(scores)

    output = attn @ value

    output = output.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)
    output = output @ W_o
    return output