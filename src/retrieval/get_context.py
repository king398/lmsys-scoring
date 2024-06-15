import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import ast
from tqdm import tqdm

model_path = 'BAAI/bge-base-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda:1")
df = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv')
prompt_df = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/open_hermes_responses.csv')
embeddings = np.load('/home/mithil/PycharmProjects/lmsys-scoring/data/open_hermes_embeddings.npy')
index = faiss.read_index("/home/mithil/PycharmProjects/lmsys-scoring/src/retrieval/knn.index",
                         faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []


df['prompt'] = df['prompt'].apply(string_to_list)
all_prompts_values = []
for i in df['prompt']:
    all_prompts_values.extend(i)

batch_size = 128
prompt_embeddings = None

# Compute embeddings for prompts in batches
for idx in tqdm(range(0, len(all_prompts_values), batch_size)):
    prompts = all_prompts_values[idx:idx + batch_size]

    batch_dict = tokenizer(prompts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    for k, v in batch_dict.items():
        batch_dict[k] = v.to('cuda:1')
    with torch.inference_mode():
        trial_embedding = model(**batch_dict).last_hidden_state[:, 0]
    if prompt_embeddings is None:
        prompt_embeddings = trial_embedding
    else:
        prompt_embeddings = torch.cat((prompt_embeddings, trial_embedding), dim=0)

prompt_embeddings = torch.nn.functional.normalize(prompt_embeddings, p=2, dim=1)
prompt_embeddings = prompt_embeddings.detach().cpu().numpy().astype('float32')

# Prepare DataFrame to store similarities
prompt_similarities_df = pd.DataFrame(columns=['prompt', 'similar_prompts', 'distances','similar_responses'])
prompt_similarities_df['prompt'] = all_prompts_values

# Compute similarities in batches
for idx in tqdm(range(0, len(prompt_embeddings), batch_size)):
    batch_embeddings = prompt_embeddings[idx:idx + batch_size]
    distances, indices = index.search(batch_embeddings, 1)
    for i, (distance, index_list) in enumerate(zip(distances, indices)):
        actual_idx = idx + i
        prompt_similarities_df.at[actual_idx, 'similar_prompts'] = prompt_df['prompt'].iloc[index_list[0]]
        prompt_similarities_df.at[actual_idx, 'distances'] = distance.tolist()
        prompt_similarities_df.at[actual_idx, 'similar_responses'] = prompt_df['response'].iloc[index_list[0]]
prompt_similarities_df['max_distance'] = prompt_similarities_df['distances'].apply(lambda x: max(x))

# Save the results
prompt_similarities_df.to_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/prompt_similarities.csv', index=False,
                              encoding='utf-8', escapechar='\\', errors='replace')
