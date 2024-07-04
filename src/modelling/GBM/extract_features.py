from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss
import ast
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
from src.modelling.llm.utils import string_to_list
from torch import nn
from src.modelling.llm.model import GemmaClassifier
from src.modelling.llm.data import prepare_input

model_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-epoch-better-prompt/checkpoint-2580"
model_name = "google/gemma-2-9b-it"
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                             device_map="cuda:1",
                                             trust_remote_code=True,
                                             attn_implementation="eager", )
# model.load_adapter(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = GemmaClassifier(model,).to("cuda:1")
model.load_adapter(model_path)
# Read and process the dataset
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", encoding='utf-8')
df = df[df['fold'] == 0].reset_index(drop=True)

df['prompt'] = df['prompt'].apply(string_to_list)
df['response_a'] = df['response_a'].apply(string_to_list)
df['response_b'] = df['response_b'].apply(string_to_list)

tokenizer.pad_token = tokenizer.eos_token


class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.text = df['text']
        self.tokenizer = tokenizer
        self.label = df['label']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {"text": self.text[idx], "label": self.label[idx]}


df['text'] = df.apply(prepare_input, axis=1, args=(tokenizer,))
dataset = EvalDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)
labels = []
hidden_states_all = None
for batch in tqdm(dataloader):
    batch['text'] += f"[RESULT]:"
    inputs = tokenizer(batch['text'], return_tensors="pt", truncation=True, max_length=2560, padding="longest")
    for k, v in inputs.items():
        inputs[k] = v.to("cuda:1")

    with torch.no_grad() and torch.cuda.amp.autocast():
        outputs  = model(**inputs)
    hidden_states = outputs['hidden_states']

    if hidden_states_all is None:
        hidden_states_all = hidden_states
    else:
        hidden_states_all = torch.cat((hidden_states_all, hidden_states), 0)

    labels.append(batch['label'].numpy())

hidden_states_all = hidden_states_all.cpu().numpy()
labels = np.array(labels)
np.save("../../../data/hidden_states_validation_gemma.npy", hidden_states_all)
np.save("../../../data/labels_validation.npy", labels)