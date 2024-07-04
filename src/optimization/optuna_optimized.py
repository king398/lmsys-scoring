import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import log_loss

model_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/Meta-Llama-3-8B-Instruct-3096-2-epoch-label-smoothing/"

logits = np.load(f"{model_path}/logits.npz")['logits']

labels = pd.read_csv(f"{model_path}/oof.csv")['label']


def objective(trial):
    clip_value_max = trial.suggest_float("clip_value_max", 0.5, 1.0)
    clip_value_min = trial.suggest_float("clip_value_min", 0.0, 0.5)
    temperature = trial.suggest_float("temperature", 0.0, 2.0)
    logits_ = logits / temperature
    logits_ = torch.tensor(logits_).softmax(dim=-1).numpy().clip(clip_value_min, clip_value_max)
    loss = log_loss(labels, logits_, )
    return loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=250)
print(study.best_params)
print(study.best_value)
print(log_loss(labels, torch.tensor(logits).softmax(dim=-1).numpy()))