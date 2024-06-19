import xgboost as xgb
import numpy as np
import pandas as pd
# Load your data
train_features = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states.npy")
train_labels = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/labels.npy")
valid_features = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states_validation.npy")
valid_labels = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/labels_validation.npy")

# Create DMatrix for XGBoost
train_dataset = xgb.DMatrix(train_features, label=train_labels)
valid_dataset = xgb.DMatrix(valid_features, label=valid_labels)
# Set parameters
params = {
    "objective": "multi:softprob",
    "num_class": len(np.unique(train_labels)),  # Number of classes
    "eval_metric": "mlogloss",
    "learning_rate": 0.005,  # You can adjust the learning rate
    "max_depth": 15,  # Depth of the trees
    "subsample": 0.8,  # Fraction of data to be used for training
    "colsample_bytree": 0.9,  # Fraction of features to be used for training
    "verbosity": 1  # Suppress logs
}

# Specify validation set
evals = [(train_dataset, 'train'), (valid_dataset, 'eval')]

# Train the model
model = xgb.train(
    params,
    train_dataset,
    evals=evals,
    num_boost_round=1000,
    early_stopping_rounds=10
)

# Get the best score and iteration
best_score = model.best_score
best_iteration = model.best_iteration

print(f"Best log loss: {best_score}")
print(f"Best iteration: {best_iteration}")
valid_preds = model.predict(valid_dataset)

np.save("/home/mithil/PycharmProjects/lmsys-scoring/data/valid_preds_lgb.npy", valid_preds)
