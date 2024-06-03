from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss
import ast
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
from src.llm.utils import string_to_list

from src.llm.data import prepare_input

model_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/Meta-Llama-3-8B-Instruct-2560-2-epoch-extra-data-lmsys/"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                             device_map="cuda:1",
                                             trust_remote_code=True, attn_implementation="flash_attention_2", )
model.load_adapter(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", encoding='utf-8')
df = df[df['fold'] != 0].reset_index(drop=True)

df['prompt'] = df['prompt'].apply(string_to_list)
df['response_a'] = df['response_a'].apply(string_to_list)
df['response_b'] = df['response_b'].apply(string_to_list)

tokenizer.pad_token = tokenizer.eos_token
df['text'] = df.apply(prepare_input, axis=1, args=(tokenizer,))

