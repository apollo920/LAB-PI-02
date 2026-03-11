import numpy as np


def softmax(x):
    
    # axis=-1 pra operar na ultima dimensao (cada linha vira uma distribuicao)
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
   
    dk = Q.shape[-1]  

   
    scores = Q @ K.transpose(0, 2, 1)  

    
    scores = scores / np.sqrt(dk)

   
    weights = softmax(scores)  

    
    output = weights @ V  

    return output, weights


class SingleHeadAttention:
    """
    Self-attention com uma unica cabeca.
    Mais simples que multi-head, mas a logica central eh a mesma.
    """

    def __init__(self, d_model):
        self.d_model = d_model
        
        self.dk = d_model
        self.W_Q = np.random.randn(d_model, self.dk) * 0.1
        self.W_K = np.random.randn(d_model, self.dk) * 0.1
        self.W_V = np.random.randn(d_model, self.dk) * 0.1

    def forward(self, X):
        """
        X: tensor de entrada, shape (batch, seq, d_model)
        Retorna: output de atencao, shape (batch, seq, d_model)
        """
        
        Q = X @ self.W_Q  
        K = X @ self.W_K  
        V = X @ self.W_V  

        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights


# ----- teste rapido -----
if __name__ == "__main__":
    np.random.seed(42)

    batch_size = 1
    seq_len = 11  
    d_model = 64

    X_test = np.random.randn(batch_size, seq_len, d_model)

    attn = SingleHeadAttention(d_model)
    out, w = attn.forward(X_test)

    print(f"Input shape:   {X_test.shape}")
    print(f"Output shape:  {out.shape}")
    print(f"Weights shape: {w.shape}")
    assert out.shape == X_test.shape, "Shape da saida errado!"
    print("[OK] Attention funcionando.")