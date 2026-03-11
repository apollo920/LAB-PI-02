import numpy as np
import pandas as pd


D_MODEL = 64


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


frase = "o banco bloqueou o cartao do cliente por suspeita de fraude"
tokens = frase.split()

token_ids = [vocab[t] for t in tokens if t in vocab]

print("Vocabulario:")
print(df_vocab)
print(f"\nFrase: '{frase}'")
print(f"Token IDs: {token_ids}")
print(f"Sequencia de {len(token_ids)} tokens")

np.random.seed(42)  
vocab_size = len(vocab)
embedding_table = np.random.randn(vocab_size, D_MODEL)

print(f"\nTabela de Embeddings: {embedding_table.shape}")


X_seq = embedding_table[token_ids]  

print(f"\nTensor de entrada X:")
print(f"  Shape: {X.shape}  (batch_size, seq_len, d_model)")
print(f"  Dtype: {X.dtype}")


if __name__ == "__main__":
    assert X.shape == (1, len(token_ids), D_MODEL), "Shape errado!"
    print("\n[OK] Shape do tensor de entrada correto.")