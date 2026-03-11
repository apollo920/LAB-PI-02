import numpy as np


def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm


def add_and_norm(x, sublayer_output, eps=1e-6):
    residual = x + sublayer_output
    return layer_norm(residual, eps)


class FeedForwardNetwork:

    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model

        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        scale = np.sqrt(2.0 / (d_model + self.d_ff))
        self.W1 = np.random.randn(d_model, self.d_ff) * scale
        self.b1 = np.zeros(self.d_ff)

        scale2 = np.sqrt(2.0 / (self.d_ff + d_model))
        self.W2 = np.random.randn(self.d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x):
    
        hidden = x @ self.W1 + self.b1

        hidden = np.maximum(0, hidden)

        out = hidden @ self.W2 + self.b2

        return out


# ----- testes -----
if __name__ == "__main__":
    np.random.seed(42)

    batch_size = 1
    seq_len = 11
    d_model = 64

    x = np.random.randn(batch_size, seq_len, d_model)


    x_normed = layer_norm(x)
    print(f"LayerNorm input shape:  {x.shape}")
    print(f"LayerNorm output shape: {x_normed.shape}")

    print(f"Media pos-norm (esperado ~0): {x_normed.mean():.6f}")
    print(f"Std pos-norm  (esperado ~1): {x_normed.std():.6f}")

    dummy_sublayer = np.random.randn(batch_size, seq_len, d_model) * 0.1
    x_res = add_and_norm(x, dummy_sublayer)
    print(f"\nResidual+Norm output shape: {x_res.shape}")
    assert x_res.shape == x.shape

    ffn = FeedForwardNetwork(d_model)
    x_ffn = ffn.forward(x)
    print(f"\nFFN input shape:  {x.shape}")
    print(f"FFN output shape: {x_ffn.shape}")
    assert x_ffn.shape == x.shape

    print("\n[OK] LayerNorm, Residual e FFN funcionando.")