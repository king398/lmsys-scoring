import lightgbm as lgb
import numpy as np
import pandas as pd

# Load your data
train_features = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states.npy")
train_labels = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/labels.npy")
valid_features = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states_validation.npy")
valid_labels = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/labels_validation.npy")

# Create Dataset for LightGBM
train_dataset = lgb.Dataset(train_features, label=train_labels)
valid_dataset = lgb.Dataset(valid_features, label=valid_labels, reference=train_dataset)

# Set parameters
params = {
    "objective": "multiclass",
    "num_class": len(np.unique(train_labels)),  # Number of classes
    "metric": "multi_logloss",
    "learning_rate": 0.01,  # You can adjust the learning rate
    "subsample": 0.8,  # Fraction of data to be used for training
    "colsample_bytree": 0.9,  # Fraction of features to be used for training
    "verbosity": 1,  # Suppress logs

}

# Train the model
model = lgb.train(
    params,
    train_dataset,
    valid_sets=[train_dataset, valid_dataset],
    num_boost_round=1000,
    early_stopping_rounds=10
)

# Get the best score and iteration
best_score = model.best_score['valid_1']['multi_logloss']
best_iteration = model.best_iteration

print(f"Best log loss: {best_score}")
print(f"Best iteration: {best_iteration}")

valid_preds = model.predict(valid_features, num_iteration=best_iteration)

np.save("/home/mithil/PycharmProjects/lmsys-scoring/data/valid_preds_lgb.npy", valid_preds)
