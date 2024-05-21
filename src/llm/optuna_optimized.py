import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import log_loss

model_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/Prometheus-eval-2-epoch-2560-len/merged"

logits = np.load(f"{model_path}/logits.npy")

labels = pd.read_csv(f"{model_path}/oof.csv")['label']


def objective(trial):
    temperature = trial.suggest_float("temperature", 0.0, 2.0)
    logits_ = logits / temperature
    logits_ = torch.tensor(logits_).softmax(dim=-1).numpy()
    preds = logits_[:, [330, 365, 14628]]
    loss = log_loss(labels, preds, )
    return loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=250)
print(study.best_params)
