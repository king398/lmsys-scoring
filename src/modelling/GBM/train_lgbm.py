import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Load your data
train_features = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states.npy")
train_labels = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/labels.npy")
valid_features = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states_validation.npy")
valid_labels = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/labels_validation.npy")
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", encoding='utf-8')
# Load your text data
train_df = df[df['fold'] != 0]
valid_df = df[df['fold'] == 0]

# Create TF-IDF features
tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 7), min_df=10,
                        max_df=0.95,lowercase=False,
                        sublinear_tf=True)  # You can adjust max_features as needed
train_tfidf = tfidf.fit_transform(train_df['text'])
valid_tfidf = tfidf.transform(valid_df['text'])

# Combine TF-IDF features with existing features
train_features_combined = hstack([train_features, train_tfidf]).tocsr()
valid_features_combined = hstack([valid_features, valid_tfidf]).tocsr()

# Create Dataset for LightGBM
train_dataset = lgb.Dataset(train_features_combined, label=train_labels)
valid_dataset = lgb.Dataset(valid_features_combined, label=valid_labels, reference=train_dataset)

# Set parameters
params = {
    "objective": "multiclass",
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "metric": "multi_logloss",
    "learning_rate": 0.02,
    "extra_trees": True,
    "num_rounds": 3000,
    "reg_lambda": 1.3,
    "num_classes": 3,
    "num_leaves": 64,
    "reg_alpha": 0.1,
    "device": "cpu",
    "max_depth": 6,
    "max_bin": 128,
    "verbose": -1,
    "seed": 42,}


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
    callbacks=[callback, lgb.early_stopping(10)],
)

# Get the best score and iteration
best_score = model.best_score['valid_1']['multi_logloss']
best_iteration = model.best_iteration
print(f"\nBest log loss: {best_score}")
print(f"Best iteration: {best_iteration}")

# Make predictions
valid_preds = model.predict(valid_features_combined, num_iteration=best_iteration)
np.save("/home/mithil/PycharmProjects/lmsys-scoring/data/valid_preds_lgb_tfidf.npy", valid_preds)
