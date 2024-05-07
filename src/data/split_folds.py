import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import ast

# Load the CSV file with explicit encoding declaration
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train.csv", encoding='utf-8')

df['label'] = np.argmax(df[['winner_model_a', 'winner_model_b', 'winner_tie']].values, axis=1)
df['fold'] = -1

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []  # Handle cases where conversion fails

df['prompt'] = df['prompt'].apply(string_to_list)
df['response_a'] = df['response_a'].apply(string_to_list)
df['response_b'] = df['response_b'].apply(string_to_list)

def create_text(row):
    text = ""
    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"Prompt: {prompt} Response_A: {response_a} Response_B: {response_b} "
    return  text

df['text'] = df.apply(create_text, axis=1)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(skf.split(df, df['label'])):
    df.loc[test_index, 'fold'] = fold

df.to_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds.csv", index=False, encoding='utf-8', errors='replace')
