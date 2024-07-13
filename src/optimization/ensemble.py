import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import torch

gemma_df = pd.read_csv(
    "/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-smoothing-2560-len/oof.csv")

llama_preds = torch.softmax(torch.tensor(np.load(
    "/home/mithil/PycharmProjects/lmsys-scoring/models/Meta-Llama-3-8B-Instruct-3096-2-epoch-label-smoothing/logits.npz")[
                                             'logits']), dim=1)
gemma_preds = gemma_df[['A', 'B', 'tie']].values

weights = np.linspace(0, 1, 1001)
best_weights = None
best_loss = float('inf')
for weight in weights:
    ensemble_preds = weight * llama_preds + (1 - weight) * gemma_preds
    loss = log_loss(gemma_df['label'], ensemble_preds)
    print(f"Weight: {weight:.3f}, Loss: {loss:.5f}")
    if loss < best_loss:
        best_loss = loss
        best_weights = weight

print(f"Best loss: {best_loss} with weight: {best_weights}")
loss = log_loss(gemma_df['label'], gemma_preds)
print('loss:', loss)
