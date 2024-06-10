import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import ast

model_path = 'Alibaba-NLP/gte-base-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, unpad_inputs=True,
                                  use_memory_efficient_attention=True, torch_dtype=torch.float16, device_map="cuda:1")
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

trial_prompt = df['prompt'][0][1]

batch_dict = tokenizer(trial_prompt, max_length=8192, padding=True, truncation=True, return_tensors='pt')
for k, v in batch_dict.items():
    batch_dict[k] = v.to('cuda:1')
with torch.inference_mode():
    trial_embedding = model(**batch_dict).last_hidden_state[:, 0]
trial_embedding = torch.nn.functional.normalize(trial_embedding, p=2, dim=1)
trial_embedding = trial_embedding.detach().cpu().numpy().astype('float32')
distances, indices = index.search(trial_embedding, 5)
print(indices)
