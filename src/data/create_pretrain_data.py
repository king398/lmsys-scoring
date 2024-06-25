import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import ast
import transformers

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Theta-Llama-3-8B')

# Load the CSV file with explicit encoding declaration
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/output_70b_batch.csv", encoding='utf-8')

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
    text = f"""Please analyze the conversation below between a human and two language models which give both respectively give the response ###Response A and ###Response B. The models are each asked to respond to the same prompts which is indicated by ###Instruction:. 
After reviewing the responses from both models, please determine which is the  better responses overall - Response_a, Response_b, or was it a tie? """

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"""
###Instruction:: {prompt} 
###Response A: {response_a} 
###Response B: {response_b}"""
    messages = [
        {"role": "user", "content": text},
        {'role': "assistant", "content": f"[RESULT]: "}
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return text


df['text'] = df.apply(create_text, axis=1)
df.to_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama_generated_text_pretrained.csv", index=False,errors='replace')