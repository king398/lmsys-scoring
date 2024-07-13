from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.modelling.llm.utils import string_to_list
from src.modelling.llm.model import GemmaClassifier
from src.modelling.llm.data import prepare_input

eval = False
model_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-smoothing-2560-len"
model_name = "google/gemma-2-9b-it"
# Load model and tokenizer
device = "cuda:1"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                             device_map=device,
                                             trust_remote_code=True,
                                             attn_implementation="eager", )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = GemmaClassifier(model, return_features=True).to("cuda:1")
model.load_adapter(model_path)
# Read and process the dataset
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", encoding='utf-8')
if eval:
    df = df[df['fold'] == 0].reset_index(drop=True)
else:
    df = df[df['fold'] != 0].reset_index(drop=True)
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
    inputs = tokenizer(batch['text'], return_tensors="pt", truncation=True, max_length=2560, padding="longest")
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad() and torch.cuda.amp.autocast():
        outputs = model(**inputs)
    hidden_states = outputs['features']

    if hidden_states_all is None:
        hidden_states_all = hidden_states
    else:
        hidden_states_all = torch.cat((hidden_states_all, hidden_states), 0)

    labels.append(batch['label'].numpy())

hidden_states_all = hidden_states_all.type(torch.float16).cpu().numpy()
labels = np.array(labels)
np.save("../../../data/hidden_states_train_gemma.npy", hidden_states_all)
np.save("../../../data/labels_train.npy", labels)
