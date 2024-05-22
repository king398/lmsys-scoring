import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import ast
import transformers
from datasets import load_dataset
import uuid

tokenizer = transformers.AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
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
    text = """Please analyze the conversation below between a human and two language models which give both respectively give the response ###Response A and ###Response B. The models are each asked to respond to the same prompts which is indicated by ###Instruction:. 
After reviewing the responses from both models, please determine which is the  better responses overall - Response_a, Response_b, or was it a tie? Respond with only a single word after [RESULT]: . Either "A" if ###Response A was better, "B" if ###Response B was better, or "tie" if their responses were equally good or bad"""

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"""
###Instruction:: {prompt} 
###Response A: {response_a} 
###Response B: {response_b}"""
    messages = [
        {"role": "user", "content": text},
        {'role': "assistant", "content": f"[RESULT]: {label_to_response[row['label']]} "}
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False
                                         )
    return text


def create_extra_data(row):
    text = """Please analyze the conversation below between a human and two language models which give both respectively give the response ###Response A and ###Response B. The models are each asked to respond to the same prompts which is indicated by ###Instruction:. 
After reviewing the responses from both models, please determine which is the  better responses overall - Response_a, Response_b, or was it a tie? Respond with only a single word after [RESULT]: . Either "A" if ###Response A was better, "B" if ###Response B was better, or "tie" if their responses were equally good or bad"""
    text += f"""
###Instruction:: {row['instruction']}
###Response A: {row['orig_response_A']}
###Response B: {row['orig_response_B']}"""
    messages = [
        {"role": "user", "content": text},
        {'role': "assistant", "content": f"[RESULT]: {row['orig_preference']} "}
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return text


df['text'] = df.apply(create_text, axis=1)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(skf.split(df, df['label'])):
    df.loc[test_index, 'fold'] = fold

preference_collection = load_dataset('prometheus-eval/Preference-Collection')['train']
# to pandas df
extra_data = preference_collection.to_pandas()
extra_data['fold'] = -1
df = df[['text', 'label', 'fold', 'id']]
extra_data['id'] = extra_data.apply(lambda x: str(uuid.uuid4()), axis=1)
extra_data.drop_duplicates(subset=['orig_response_A', 'orig_response_B'], inplace=True)
extra_data = extra_data.sample(frac=0.25, random_state=42).reset_index(drop=True)

extra_data['text'] = extra_data.apply(create_extra_data, axis=1)
extra_data['label'] = extra_data['orig_preference'].map({'A': 0, 'B': 1, 'tie': 2})
df = pd.concat([df, extra_data[['text', 'label', 'fold', 'id']]], ignore_index=True)

df.to_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_extra_data.csv", index=False, encoding='utf-8',
          errors="ignore")
