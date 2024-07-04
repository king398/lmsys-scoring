import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import ast
import transformers

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

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

label_to_response = {0: 'A', 1: 'B', 2: 'tie'}


def create_text(row):
    text = f""""""

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"Instruction:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}\n\n"
    messages = [
        {"role": "user", "content": text},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return text

df['text'] = df.apply(create_text, axis=1)

# Add a column indicating the row is not swapped


skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(skf.split(df, df['label'])):
    df.loc[test_index, 'fold'] = fold



df.to_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", index=False,
          encoding='utf-8', errors='replace')

