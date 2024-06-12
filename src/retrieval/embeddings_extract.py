import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

model_path = 'BAAI/bge-base-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,  torch_dtype=torch.float16, device_map="cuda:1")

data_df = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/open_hermes_responses.csv')

all_embeddings = None

prompts = data_df['prompt'].tolist()
batch_size = 256
for i in tqdm(range(0, len(prompts), batch_size)):
    batch_prompt = prompts[i:i + batch_size]
    batch_dict = tokenizer(batch_prompt, max_length=512, padding=True, truncation=True, return_tensors='pt')

    for k, v in batch_dict.items():
        batch_dict[k] = v.to('cuda:1')
    with torch.inference_mode():
        embeddings = model(**batch_dict).last_hidden_state[:, 0]
    if all_embeddings is None:
        all_embeddings = embeddings
    else:
        all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)

all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
all_embeddings = all_embeddings.detach().cpu().numpy()
np.save('/home/mithil/PycharmProjects/lmsys-scoring/data/open_hermes_embeddings.npy', all_embeddings)
