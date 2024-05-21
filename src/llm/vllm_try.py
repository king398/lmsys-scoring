import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import ast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from vllm.lora.request import LoRARequest

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, top_k=1, max_tokens=5, logprobs=32000,
                                 )
llm = LLM(model="prometheus-eval/prometheus-7b-v2.0", max_model_len=16000, enable_lora=True,
          max_lora_rank=64)
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
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
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
dataloader = DataLoader(dataset, batch_size=2, num_workers=8, pin_memory=True)
predictions = []
labels = []
for batch in tqdm(dataloader):
    batch['text'] = [text + "[RESULT]: " for text in batch['text']]
    outputs = llm.generate(batch['text'], sampling_params, use_tqdm=False, lora_request=LoRARequest("main_adapter", 1,
                                                                                                    "/home/mithil/PycharmProjects/lmsys-scoring/models/Promethus-eval-1-epoch/checkpoint-1347"))
    for i in outputs:
        print(i.text)
        i = i.outputs[0]
        logprob_dict = i.logprobs[0]
        # sort the dict by its keys
        sorted_logprob_dict = {k: logprob_dict[k] for k in sorted(logprob_dict)}
        logprobs = torch.Tensor(list(sorted_logprob_dict.values())).softmax(dim=-1)
        predictions.append(logprobs[[330, 365, 14628]].numpy().tolist())
    labels.append(batch['label'])


from sklearn.metrics import log_loss
loss = log_loss(labels, predictions)
print(loss)