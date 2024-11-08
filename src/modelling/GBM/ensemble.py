import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

df = pd.read_csv(
    "/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-smoothing-2560-len/oof.csv")
lgbm_preds = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/valid_preds_lgb_tfidf.npy")
llama_preds = df[['A', 'B', 'tie']].values

weights = np.linspace(0, 1, 1001)
best_weights = None
best_loss = float('inf')
for weight in weights:
    ensemble_preds = weight * lgbm_preds + (1 - weight) * llama_preds
    loss = log_loss(df['label'], ensemble_preds)
    print(f"Weight: {weight:.3f}, Loss: {loss:.5f}")
    if loss < best_loss:
        best_loss = loss
        best_weights = weight

print(f"Best weight: {best_weights:.3f}, Best loss: {best_loss:.5f}")
print(f" Baseline loss: {log_loss(df['label'], llama_preds)}")