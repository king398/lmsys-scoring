from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm

tqdm.pandas()
import ast

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16,
                                             device_map="cuda:1")
from torch.utils.data import DataLoader, Dataset

model.load_adapter("/home/mithil/PycharmProjects/lmsys-scoring/models/llama-3-8b")
tokenizer = AutoTokenizer.from_pretrained("/home/mithil/PycharmProjects/lmsys-scoring/models/llama-3-8b")
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", encoding='utf-8')
df = df[df['fold'] == 0].reset_index(drop=True)


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []  # Handle cases where conversion fails


df['prompt'] = df['prompt'].apply(string_to_list)
df['response_a'] = df['response_a'].apply(string_to_list)
df['response_b'] = df['response_b'].apply(string_to_list)

tokenizer.pad_token = tokenizer.eos_token
def prepare_input(row):
    text = """Please analyze the conversation below between a human and two language models (model_a and model_b). The models are each asked to respond to the same prompts which is indicated by Prompt. 
After reviewing the responses from both models, please determine which model provided better responses overall - model_a, model_b, or was it a tie? Respond with only a single word after <result>: either "A" if model_a was better, "B" if model_b was better, or "tie" if their responses were equally good"""

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"""
    prompt: {prompt} 
    model_a: {response_a} 
    model_b: {response_b}"""
    text += f""" <result>:"""
    # truncate to 3050 tokens
    text = tokenizer.decode(tokenizer(text, return_tensors="pt", truncation=True, max_length=3000)['input_ids'][0])
    messages = [{"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": text},
                ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False
                                         )
    return text


class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.text = df['text']
        self.tokenizer = tokenizer
        self.label = df['label']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {"text": self.text[idx], "label": self.label[idx]}


df['text'] = df.apply(prepare_input, axis=1)
predictions = []
labels = []
dataset = EvalDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True)
for batch in tqdm(dataset):
    inputs = tokenizer(batch['text'], return_tensors="pt", truncation=True, max_length=3096, padding=True)
    for k, v in inputs.items():
        inputs[k] = v.to("cuda:1")
    outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, output_scores=True,
                             return_dict_in_generate=True,pad_token_id=tokenizer.eos_token_id)['scores'][0].softmax(dim=1)
    predictions.extend(outputs[:, [32, 33, 49831]].detach().cpu().numpy().tolist())
    labels.append(batch['label'])


print(predictions)
from sklearn.metrics import log_loss
print(log_loss(labels, predictions))