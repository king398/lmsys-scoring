import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss
import ast
from torch.utils.data import DataLoader
from src.llm.utils import string_to_list

from src.llm.data import prepare_input, EvalDataset

model_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/Meta-Llama-3-8B-Instruct-2560-2-epoch-extra-data-lmsys/"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                             device_map="cuda:0",
                                             trust_remote_code=True, attn_implementation="flash_attention_2", )
model.load_adapter(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", encoding='utf-8')
df = df[df['fold'] == 0].reset_index(drop=True)[:10000]

df['prompt'] = df['prompt'].apply(string_to_list)
df['response_a'] = df['response_a'].apply(string_to_list)
df['response_b'] = df['response_b'].apply(string_to_list)

tokenizer.pad_token = tokenizer.eos_token
df['text'] = df.apply(prepare_input, axis=1, args=(tokenizer,))

dataset = EvalDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


hidden_states_all = None
labels = []
for batch in tqdm(dataloader):
    batch['text'] += f"[RESULT]:"
    inputs = tokenizer(batch['text'], return_tensors="pt", truncation=True, max_length=2560, padding="longest")
    for k, v in inputs.items():
        inputs[k] = v.to("cuda:0")

    outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, output_hidden_states=True,
                             return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
    hidden_states = outputs['hidden_states'][0][-1]
    hidden_states = mean_pooling(hidden_states, inputs['attention_mask']).detach().cpu()
    if hidden_states_all is None:
        hidden_states_all = hidden_states
    else:
        hidden_states_all = torch.cat((hidden_states_all, hidden_states), 0)
    labels.append(batch['label'].numpy())


hidden_states_all = hidden_states_all.numpy()
np.savez_compressed("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states_eval.npz",
                    hidden_states_all=hidden_states_all, labels=labels)
