import vllm
import torch
import pandas as pd
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    gemma_dir = '/kaggle/input/gemma-finetuning-all/gemma-2-9b/'
    max_length = 1536


cfg = Config()

llm = vllm.LLM(
    cfg.gemma_dir,
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=cfg.max_length,
    tensor_parallel_size=2,
    distributed_executor_backend="mp",

)
tokenizer = llm.get_tokenizer()

test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')[:25000]


def process_text(text: str) -> str:
    return " ".join(eval(text, {"null": ""}))


test.loc[:, 'prompt'] = test['prompt'].apply(process_text)
test.loc[:, 'response_a'] = test['response_a'].apply(process_text)
test.loc[:, 'response_b'] = test['response_b'].apply(process_text)


def tokenize(
        prompt, response_a, response_b
):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]
    text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
    return text


tokenizer.add_eos_token = True
tokenizer.padding_side = "right"
data = pd.DataFrame()
data['text'] = tokenize(test['prompt'], test['response_a'], test['response_b'])
data['input_ids'] = data['text'].apply(
    lambda x: tokenizer(x, truncation=True, max_length=cfg.max_length, padding=False)['input_ids'])
data['text'] = data['input_ids'].apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))
data['len'] = data['input_ids'].apply(len)
data = data.sort_values('len', ascending=False)
data['id'] = test['id']

aug_data = pd.DataFrame()
aug_data['text'] = tokenize(test['prompt'], test['response_b'], test['response_a'])
aug_data['input_ids'] = aug_data['text'].apply(
    lambda x: tokenizer(x, truncation=True, max_length=cfg.max_length, padding=False)['input_ids'])
aug_data['len'] = aug_data['input_ids'].apply(len)
aug_data = aug_data.sort_values('len', ascending=False)
aug_data['text'] = aug_data['input_ids'].apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))
aug_data['id'] = test['id']
data_embeddings = []
embeddings = llm.encode(
    data['text'].values, vllm.SamplingParams(), use_tqdm=True
)

for i, embed in enumerate(embeddings):
    data_embeddings.append(embed.outputs.embedding)

data_embeddings = np.array(data_embeddings)
data_embeddings = np.array(torch.tensor(data_embeddings).softmax(dim=-1))
aug_data_embeddings = []
embeddings = llm.encode(
    aug_data['text'].values, vllm.SamplingParams(), use_tqdm=True
)
for i, embed in enumerate(embeddings):
    aug_data_embeddings.append(embed.outputs.embedding)
aug_data_embeddings = np.array(aug_data_embeddings)
aug_data_embeddings = np.array(torch.tensor(aug_data_embeddings).softmax(dim=-1))

probs = []
for i in range(len(data_embeddings)):
    aug_data_probs = aug_data_embeddings[i]
    aug_data_probs[0], aug_data_probs[1] = aug_data_probs[1], aug_data_probs[0]
    data_probs = data_embeddings[i]
    prob = data_probs * 0.5 + aug_data_probs * 0.5
    probs.append(prob)
submission = pd.DataFrame(probs, columns=['winner_model_a', 'winner_model_b', 'winner_tie'])
submission['id'] = data['id']
submission.to_csv('submission.csv', index=False)
