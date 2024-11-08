from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel, BitsAndBytesConfig
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss
import ast
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
from utils import string_to_list
from torch import nn
from src.modelling.llm.data import prepare_input

model_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/llama-3096-8b-3.1-no-smooth"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                             device_map="cuda:1",
                                             trust_remote_code=True,
                                             attn_implementation="eager", )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class GemmaClassifier(LlamaPreTrainedModel):
    def __init__(self, model, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3).to("cuda:1")

    def forward(self, inputs):
        outputs = self.model(**inputs, return_dict=True)
        hidden_states = outputs['logits']
        hidden_states = mean_pooling(hidden_states, inputs['attention_mask']).type(torch.float16)

        return {"logits": self.linear_head(hidden_states)}


model = GemmaClassifier(model).to("cuda:1")
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

# Implement reverse truncation

predictions_all = []
labels = []
logits_all = None
for batch in tqdm(dataloader):
    inputs = tokenizer.encode_plus(batch['text'], max_length=2560, truncation=True, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to("cuda:1")
    with torch.no_grad() and torch.cuda.amp.autocast():
        outputs = model(inputs)
        if logits_all is None:
            logits_all = outputs['logits']
        else:
            logits_all = torch.cat((logits_all, outputs['logits']), dim=0)
        predictions = outputs['logits'].softmax(dim=-1)
        predictions_all.append(predictions)
        labels.append(batch['label'])

predictions_all = torch.cat(predictions_all, dim=0).cpu().numpy()
labels = torch.tensor(labels).cpu().numpy()
log_loss_all = log_loss(labels, predictions_all)
print(f"Log loss: {log_loss_all}")
oof_df = pd.DataFrame(predictions_all, columns=["A", "B", "tie"])
oof_df["label"] = df["label"]
oof_df['id'] = df['id']
log_losses = []

for i in range(len(df)):
    log_loss_value = log_loss([labels[i]], [predictions_all[i]], labels=[0, 1, 2])
    log_losses.append(log_loss_value)

oof_df["log_loss"] = log_losses

oof_df['perplexity'] = oof_df.apply(lambda x: math.e ** x["log_loss"], axis=1)
oof_df.to_csv(f"{model_path}/oof.csv", index=False)
np.savez_compressed(f"{model_path}/logits.npz", logits=logits_all.detach().cpu().numpy())
