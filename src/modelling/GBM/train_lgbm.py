import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

base_path = "/home/mithil/PycharmProjects/lmsys-scoring/data/embeddings"
# Load your data
train_features = np.load(f"{base_path}/gemma-2-9b-it-bnb-4bit_fold0_1024_full_train.npy")
train_labels = np.load(f"{base_path}/labels_train.npy")
valid_features = np.load(f"{base_path}/gemma-2-9b-it-bnb-4bit_fold0_1024_full_valid.npy")
valid_labels = np.load(f"{base_path}/labels_valid.npy")
train_df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", encoding='utf-8')
valid_df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/lmsys-33k-deduplicated.csv", encoding='utf-8')
# Load your text data
valid_df = valid_df[
    valid_df['model_a'].isin(train_df['model_a']) & valid_df['model_b'].isin(train_df['model_b'])].reset_index(
    drop=True)

# Create TF-IDF features

# Create Dataset for LightGBM
# merge train and valid datasets
train_features = np.concatenate([train_features, valid_features], axis=0)
train_labels = np.concatenate([train_labels, valid_labels], axis=0)
train_dataset = lgb.Dataset(train_features, label=train_labels)
valid_dataset = lgb.Dataset(valid_features, label=valid_labels, reference=train_dataset)

# Set parameters
params = {
    "objective": "multiclass",
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "extra_trees": True,
    "num_rounds": 1500,
    "reg_lambda": 1.3,
    "num_classes": 3,
    "num_leaves": 64,
    "reg_alpha": 0.1,
    "device": "cpu",
    "max_depth": 6,
    "max_bin": 128,
    "verbose": -1,
    "seed": 42, }


# Define a callback function to print log loss at each iteration
def callback(env):
    if env.evaluation_result_list:
        train_logloss = env.evaluation_result_list[0][2]
        valid_logloss = env.evaluation_result_list[1][2]
        print(f"Iteration {env.iteration}, Train Log Loss: {train_logloss:.6f}, Valid Log Loss: {valid_logloss:.6f}")


# Train the model
model = lgb.train(
    params,
    train_dataset,
    valid_sets=[train_dataset, valid_dataset],
    num_boost_round=1000,
    callbacks=[callback, lgb.early_stopping(50)],

)

# Get the best score and iteration
best_score = model.best_score['valid_1']['multi_logloss']
best_iteration = model.best_iteration
print(f"\nBest log loss: {best_score}")
print(f"Best iteration: {best_iteration}")

# Make predictions
valid_preds = model.predict(valid_features, num_iteration=best_iteration)
np.save("/home/mithil/PycharmProjects/lmsys-scoring/data/valid_preds_lgb.npy", valid_preds)
model.save_model("/home/mithil/PycharmProjects/lmsys-scoring/models/lgbm__model.txt")
# save the train_preds
train_preds = model.predict(train_features, num_iteration=best_iteration)
train_preds = pd.DataFrame(train_preds, columns=['A_log', 'B_log', 'tie_log'])
train_preds['id'] = list(train_df['id']) + list(valid_df['id'])
train_preds.to_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_preds_lgb.csv",
                   index=False)
