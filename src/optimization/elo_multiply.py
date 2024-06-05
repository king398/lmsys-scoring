import numpy as np
import pandas as pd
import json
from sklearn.metrics import log_loss
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds.csv")
oof_csv = pd.read_csv(
    "/home/mithil/PycharmProjects/lmsys-scoring/models/Meta-Llama-3-8B-Instruct-2560-2-epoch-extra-data-lmsys/oof.csv")
id_model_a_dict = dict(zip(df['id'], df['model_a']))
id_model_b_dict = dict(zip(df['id'], df['model_b']))
oof_csv['model_a'] = oof_csv['id'].map(id_model_a_dict)
oof_csv['model_b'] = oof_csv['id'].map(id_model_b_dict)

with open("/home/mithil/PycharmProjects/lmsys-scoring/data/elo_weights.json", "r") as f:
    model_combination_weight_dict = json.load(f)
preds = []
labels = []
for index, row in oof_csv.iterrows():
    model_combination_a = f"{row['model_a']} {row['model_b']}"
    model_combination_b = f"{row['model_b']} {row['model_a']}"
    if model_combination_a in model_combination_weight_dict:
        weights = model_combination_weight_dict[model_combination_a]
    else:
        weights = model_combination_weight_dict[model_combination_b]
        weights = [weights[1], weights[0], weights[2]]
    pred = row[["A", "B", 'tie']].values
    scale_factor = np.sum(pred) / np.sum(weights * pred(2))
    pred = weights * pred * scale_factor
    preds.append(pred)
    labels.append(row['label'])

print(log_loss(y_true=labels, y_pred=preds))
