from llama_cpp import Llama
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import ast
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

llm = Llama(
    model_path="/home/mithil/PycharmProjects/lmsys-scoring/models/prometheus-7b-v2.0.Q8_0.gguf",
    logits_all=True,
    n_ctx=32000,
    n_gpu_layers=-1,
    offload_kqv=True,
    n_batch=8000,
    verbose=False,

)
tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0", trust_remote_code=True)

# Read and process the dataset
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


# Prepare the input text
def prepare_input(row):
    text = """Please analyze the conversation below between a human and two language models which give both respectively give the response ###Response A and ###Response B. The models are each asked to respond to the same prompts which is indicated by ###Instruction:. 
After reviewing the responses from both models, please determine which is the better response overall - Response_a, Response_b, or was it a tie? Respond with only a single word after [RESULT]: . Either "A" if ###Response A was better, "B" if ###Response B was better, or "tie" if their responses were equally good or bad"""

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"""
###Instruction:: {prompt} 
###Response A: {response_a} 
###Response B: {response_b}"""
    messages = [
        {"role": "user", "content": text},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
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
dataset = EvalDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)
predictions = []
labels = []
for batch in tqdm(dataloader):
    batch['text'] += f"[RESULT]:"
    # encode the prompt as utf-8
    try:
        output = llm.create_completion(batch['text'], echo=False, max_tokens=2, logprobs=32000, temperature=0.0,
                                       top_k=1, )
        all_logprob = torch.tensor(list(output['choices'][0]['logprobs']['top_logprobs'][0].values())).softmax(dim=-1)
        all_keys = list(output['choices'][0]['logprobs']['top_logprobs'][0].keys())
        label_keys = ' A', ' B', ' tie'
        index_keys = []
        for i, token in enumerate(all_keys):
            if token in label_keys:
                index_keys.append(i)
        logprob = all_logprob[index_keys]
        predictions.append(logprob)
        labels.append(batch['label'])
        print(logprob)
    except Exception as e:
        print(e)
        pass

from sklearn.metrics import log_loss

log_loss = log_loss(labels, predictions)
print(log_loss)
