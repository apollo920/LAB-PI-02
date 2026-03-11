import numpy as np
import pandas as pd

from attention import SingleHeadAttention
from sublayers import add_and_norm, FeedForwardNetwork, layer_norm


class EncoderLayer:

    def __init__(self, d_model, d_ff=None):
        self.attention = SingleHeadAttention(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, X):
        X_att, _ = self.attention.forward(X)
        X_norm1 = add_and_norm(X, X_att)

        X_ffn = self.ffn.forward(X_norm1)
        X_out = add_and_norm(X_norm1, X_ffn)

        return X_out


class TransformerEncoder:

    def __init__(self, d_model=64, n_layers=6, d_ff=None):
        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = [EncoderLayer(d_model, d_ff) for _ in range(n_layers)]

    def forward(self, X):
        print(f"\n[Encoder] Iniciando forward pass")
        print(f"[Encoder] Shape entrada: {X.shape}")

        for i, layer in enumerate(self.layers):
            X = layer.forward(X)
            print(f"[Encoder] Camada {i+1}/6 -> shape: {X.shape}")

        print(f"[Encoder] Shape saida (Z): {X.shape}")
        return X 

def build_vocab_and_embeddings(d_model):
    """Monta vocabulario, DataFrame e tabela de embeddings."""
    vocab = {
        "o": 0,
        "banco": 1,
        "bloqueou": 2,
        "cartao": 3,
        "do": 4,
        "cliente": 5,
        "por": 6,
        "suspeita": 7,
        "de": 8,
        "fraude": 9,
    }

    df_vocab = pd.DataFrame(list(vocab.items()), columns=["palavra", "id"])

    np.random.seed(42)
    vocab_size = len(vocab)
    embedding_table = np.random.randn(vocab_size, d_model)

    return vocab, df_vocab, embedding_table


def tokenize(frase, vocab):
    """Converte frase em lista de IDs inteiros."""
    tokens = frase.lower().split()
    ids = [vocab[t] for t in tokens if t in vocab]
    return ids


def get_input_tensor(token_ids, embedding_table):
    """
    Faz o lookup dos embeddings e adiciona dimensao de batch.
    Retorna X de shape (1, seq_len, d_model).
    """
    X_seq = embedding_table[token_ids]          
    X = X_seq[np.newaxis, :, :]                 
    return X



if __name__ == "__main__":

    D_MODEL = 64
    N_LAYERS = 6

    frase = "o banco bloqueou o cartao do cliente por suspeita de fraude"

    # --- Passo 1: dados ---
    vocab, df_vocab, emb_table = build_vocab_and_embeddings(D_MODEL)

    print("=" * 55)
    print("VOCABULARIO")
    print("=" * 55)
    print(df_vocab.to_string(index=False))

    token_ids = tokenize(frase, vocab)
    print(f"\nFrase : '{frase}'")
    print(f"Tokens: {frase.split()}")
    print(f"IDs   : {token_ids}")

    X = get_input_tensor(token_ids, emb_table)
    print(f"\nTensor X (entrada): shape {X.shape}")

    # --- Passo 2 + 3: encoder ---
    print("\n" + "=" * 55)
    print("ENCODER - FORWARD PASS")
    print("=" * 55)

    encoder = TransformerEncoder(d_model=D_MODEL, n_layers=N_LAYERS)
    Z = encoder.forward(X)

    # --- Validacao de sanidade ---
    print("\n" + "=" * 55)
    print("VALIDACAO DE SANIDADE")
    print("=" * 55)
    assert Z.shape == X.shape, f"Shape errado! Esperado {X.shape}, obtido {Z.shape}"
    print(f"Shape entrada X : {X.shape}")
    print(f"Shape saida   Z : {Z.shape}")
    print(f"Shapes identicos: OK")

    # pequena amostra dos vetores de saida
    print(f"\nPrimeiro token - vetor Z[0][0] (primeiros 8 valores):")
    print(np.round(Z[0][0][:8], 4))

    print(f"\nUltimo token  - vetor Z[0][-1] (primeiros 8 valores):")
    print(np.round(Z[0][-1][:8], 4))

    print("\n[OK] Pipeline completo executado com sucesso!")