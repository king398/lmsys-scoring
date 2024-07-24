import pandas as pd
import numpy as np
import optuna
import torch
from sklearn.metrics import log_loss
from transformers import AutoTokenizer

# Load and preprocess data
df = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv')
pred_df = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/trial.csv')
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

df['len'] = df['text'].apply(lambda x: len(tokenizer(x)['input_ids']))
print("total tokens:", df['len'].sum())
id_df_dict = dict(zip(df['id'], df['len']))
pred_df['len'] = pred_df['id'].map(id_df_dict)
pred_df['len_bin'] = pd.qcut(pred_df['len'], q=50)

# Load model outputs
model_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-smoothing-2560-len/"
logits = np.load(f"{model_path}/logits.npz")['logits']
labels = pd.read_csv(f"{model_path}/oof.csv")['label']

# Assign bin to each sample
bin_edges = pred_df['len_bin'].cat.categories
sample_bins = pd.cut(pred_df['len'], bins=bin_edges)

def objective(trial):
    # Create a temperature for each bin
    temperatures = {
        bin_: trial.suggest_float(f"temperature_{i}", 0.5, 1.5)
        for i, bin_ in enumerate(bin_edges)
    }

    # Apply temperature and clipping for each bin
    adjusted_logits = []
    for bin_, logits_bin in zip(sample_bins, logits):
        temp = temperatures[bin_]
        logits_bin = logits_bin / temp
        probs = torch.tensor(logits_bin).softmax(dim=-1).numpy()
        adjusted_logits.append(probs)

    adjusted_logits = np.vstack(adjusted_logits)
    loss = log_loss(labels, adjusted_logits)
    return loss

# Create and run the study
study = optuna.create_study(direction="minimize",)
study.optimize(objective, n_trials=1,show_progress_bar=True)

# Print results
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)

# Calculate and print the baseline loss
baseline_loss = log_loss(labels, torch.tensor(logits).softmax(dim=-1).numpy())
print("Baseline loss:", baseline_loss)

# Calculate average log loss for each bin using optimized parameters
best_params = study.best_params

binned_losses = []
bin_mean = []
for bin_ in bin_edges:
    bin_mask = (sample_bins == bin_)
    bin_logits = logits[bin_mask]
    bin_labels = labels[bin_mask]

    temp = 1
    adjusted_logits = bin_logits / temp
    probs = torch.tensor(adjusted_logits).softmax(dim=-1).numpy()

    bin_loss = log_loss(bin_labels, probs)
    binned_losses.append( bin_loss)
    bin_mean.append(bin_.mid)

# Print the results
print("\nAverage Log Loss for each length bin after optimization:")
for bin_, loss in binned_losses:
    print(f"Bin {bin_}: Average Log Loss = {loss:.4f}")