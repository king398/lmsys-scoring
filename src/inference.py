# %%writefile gemma_infer.py
import torch
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding, \
    Gemma2PreTrainedModel
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from threading import Thread, Lock
import ast
from torch import nn
import numpy as np

torch.backends.cuda.enable_mem_efficient_sdp(True)
lock = Lock()

# Updated length bins and temperature values
LENGTH_BINS = [
    (33.999, 133.0), (133.0, 196.0), (196.0, 263.0), (263.0, 333.0),
    (333.0, 406.0), (406.0, 481.0), (481.0, 551.0), (551.0, 623.0),
    (623.0, 703.0), (703.0, 782.0), (782.0, 878.0), (878.0, 999.0),
    (999.0, 1194.0), (1194.0, 1639.0), (1639.0, 12162.0)
]

TEMPERATURE_VALUES = [
    0.893589034729071, 1.0004855565620758, 1.1263487420741431, 1.1581934258630444,
    1.04433388339559, 1.1345412861901027, 1.3728830890449788, 1.1908131852105728,
    1.3053187821060233, 1.2316662831371683, 1.1651808638637695, 1.1928528170888468,
    1.1644255873393226, 1.210707490377071, 1.4036479861831515
]


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []


def get_temperature(length):
    for (bin_start, bin_end), temp in zip(LENGTH_BINS, TEMPERATURE_VALUES):
        if bin_start < length <= bin_end:
            return temp
    return 1.0  # Default temperature if no bin matches


def prepare_input(row, tokenizer):
    text = f""""""

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"Instruction:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}\n\n"
    messages = [
        {"role": "user", "content": text},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

    inputs = tokenizer.encode_plus(text, return_tensors=None, truncation=True, max_length=cfg['max_len'])
    input_length = len(inputs['input_ids'])
    temperature = get_temperature(input_length)

    return inputs['input_ids'], inputs['attention_mask'], input_length, temperature


class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.input_ids = df['input_ids']
        self.attention_mask = df['attention_mask']
        self.temperature = df['temperature']
        self.id = df['id']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if idx >= len(self.input_ids):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self.input_ids)}")
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        temperature = self.temperature[idx]

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'temperature': temperature,
                'id': self.id[idx]}


def run_inference(dataset, tokenizer, model, device, results, index, ids_list):
    predictions = []
    ids = []
    for batch in tqdm(
            DataLoader(dataset, batch_size=cfg['batch_size'], collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
                       pin_memory=True, prefetch_factor=8, num_workers=4)):
        ids.extend(batch.pop('id'))
        temperatures = batch.pop('temperature').to(device)
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad() and torch.cuda.amp.autocast():
            outputs = model(batch)

        # Apply temperature scaling before softmax
        logits = outputs['logits'] / temperatures.unsqueeze(1)
        batch_predictions = logits.softmax(dim=-1).detach().cpu().numpy().tolist()

        predictions.extend(batch_predictions)
        del batch, outputs
        gc.collect()

    with lock:
        results[index] = predictions
        ids_list[index] = ids


class GemmaClassifier(Gemma2PreTrainedModel):
    def __init__(self, model, device, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3).to(device)

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, tensors, **kwargs):
        outputs = self.model(**tensors, return_dict=True)
        hidden_states = outputs['logits']
        hidden_states = self.mean_pooling(hidden_states, tensors['attention_mask']).type(torch.float16)
        logits = self.linear_head(hidden_states)
        return {"logits": logits}


def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], trust_remote_code=True)
    test_csv = pd.read_csv(cfg['test_csv'], encoding='utf-8')

    test_csv['prompt'] = test_csv['prompt'].apply(string_to_list)
    test_csv['response_a'] = test_csv['response_a'].apply(string_to_list)
    test_csv['response_b'] = test_csv['response_b'].apply(string_to_list)

    # Apply the prepare_input function and add new columns directly
    test_csv['input_ids'], test_csv['attention_mask'], test_csv['input_length'], test_csv['temperature'] = zip(
        *test_csv.apply(prepare_input, axis=1, args=(tokenizer,)))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False)

    model_1 = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.float16,
                                                   device_map="cuda:0", trust_remote_code=True,
                                                   quantization_config=bnb_config, attn_implementation="eager", )
    model_1 = GemmaClassifier(model_1, "cuda:0")
    model_1.load_adapter(cfg['adapter_path'])
    print(model_1)

    model_2 = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.float16,
                                                   device_map="cuda:1", trust_remote_code=True,
                                                   quantization_config=bnb_config, attn_implementation="eager", )
    model_2 = GemmaClassifier(model_2, "cuda:1")
    model_2.load_adapter(cfg['adapter_path'])

    tokenizer.pad_token = tokenizer.eos_token

    half = len(test_csv) // 2
    test_csv_1 = test_csv.iloc[:half].reset_index(drop=True)
    test_csv_2 = test_csv.iloc[half:].reset_index(drop=True)
    # sort by input length
    test_csv_1 = test_csv_1.sort_values('input_length', ascending=False).reset_index(drop=True)
    test_csv_2 = test_csv_2.sort_values('input_length', ascending=False).reset_index(drop=True)
    dataset_1 = EvalDataset(test_csv_1, tokenizer)
    dataset_2 = EvalDataset(test_csv_2, tokenizer)

    results = {}
    ids_list = {}
    t0 = Thread(target=run_inference, args=(dataset_1, tokenizer, model_1, 'cuda:0', results, 0, ids_list))
    t1 = Thread(target=run_inference, args=(dataset_2, tokenizer, model_2, 'cuda:1', results, 1, ids_list))

    t0.start()
    t1.start()

    t0.join()
    t1.join()

    predictions = results.get(0, []) + results.get(1, [])
    ids = ids_list.get(0, []) + ids_list.get(1, [])
    # make all the ids strs from tensors
    ids = [str(i.item()) for i in ids]
    submission_df = pd.DataFrame(predictions, columns=['winner_model_a', 'winner_model_b', 'winner_tie'])
    submission_df['id'] = ids
    submission_df.to_csv("submission_gemma.csv", index=False)
    torch.cuda.empty_cache()
    del model_1, model_2, tokenizer, dataset_1, dataset_2, results, ids_list, submission_df, test_csv, test_csv_1, test_csv_2
    gc.collect()


cfg = {
    "model_path": "/kaggle/input/gemma-9b-2",
    "adapter_path": "/kaggle/input/gemma-2-9b-it-smoothing-2560-len",
    "max_len": 2560,
    "test_csv": "/kaggle/input/lmsys-chatbot-arena/test.csv",
    "batch_size": 3,
}

if __name__ == "__main__":
    main(cfg)
# %%writefile llama_infer.py
import torch
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding, \
    LlamaPreTrainedModel
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from threading import Thread, Lock
import ast
from torch import nn

torch.backends.cuda.enable_mem_efficient_sdp(False)

lock = Lock()


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []


def prepare_input(row, tokenizer):
    text = """Please analyze the conversation below between a human and two language models which give both respectively give the response ###Response A and ###Response B. The models are each asked to respond to the same prompts which is indicated by ###Instruction:. 
After reviewing the responses from both models, please determine which is the better response overall - Response_a, Response_b, or was it a tie? Respond with only a single word after [RESULT]: . Either "A" if ###Response A was better, "B" if ###Response B was better, or "tie" if their responses were equally good or bad"""

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"""
###Instruction:: {prompt} 
###Response A: {response_a} 
###Response B: {response_b}"""
    text = tokenizer.decode(
        tokenizer(text, return_tensors="pt", truncation=True, max_length=3000)['input_ids'][0]
    )

    messages = [
        {"role": "user", "content": text},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return text


class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.text = df['text']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if idx >= len(self.text):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self.text)}")
        text = self.text[idx]
        text += "[RESULT]:"
        inputs = self.tokenizer.encode_plus(text, return_tensors=None, truncation=True, max_length=cfg['max_len'])

        return {**inputs}


def run_inference(dataset, tokenizer, model, device, results, index):
    predictions = []
    for batch in tqdm(
            DataLoader(dataset, batch_size=cfg['batch_size'], collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))):
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad() and torch.cuda.amp.autocast():
            outputs = model(batch)
        batch_predictions = outputs['logits'].softmax(dim=-1).detach().cpu().numpy().tolist()

        predictions.extend(batch_predictions)
        del batch, outputs
        gc.collect()

    with lock:
        results[index] = predictions


class LLamaClassifier(LlamaPreTrainedModel):
    def __init__(self, model, device, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3).to(device)

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, tensors, **kwargs):
        outputs = self.model(**tensors, return_dict=True)
        hidden_states = outputs['logits']
        hidden_states = self.mean_pooling(hidden_states, tensors['attention_mask']).type(torch.float16)

        return {"logits": self.linear_head(hidden_states)}


def main(cfg):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False)

    model_1 = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.float16,
                                                   device_map="cuda:0", trust_remote_code=True,
                                                   quantization_config=bnb_config)
    model_1 = LLamaClassifier(model_1, "cuda:0")
    model_1.load_adapter(cfg['adapter_path'])
    # model_1.gradient_checkpointing_enable()
    model_2 = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.float16,
                                                   device_map="cuda:1", trust_remote_code=True,
                                                   quantization_config=bnb_config)
    model_2 = LLamaClassifier(model_2, "cuda:1")
    model_2.load_adapter(cfg['adapter_path'])

    # model_2.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], trust_remote_code=True)
    test_csv = pd.read_csv(cfg['test_csv'], encoding='utf-8')
    test_csv['prompt'] = test_csv['prompt'].apply(string_to_list)
    test_csv['response_a'] = test_csv['response_a'].apply(string_to_list)
    test_csv['response_b'] = test_csv['response_b'].apply(string_to_list)
    test_csv['text'] = test_csv.apply(prepare_input, axis=1, args=(tokenizer,))
    tokenizer.pad_token = tokenizer.eos_token

    half = len(test_csv) // 2
    test_csv_1 = test_csv.iloc[:half].reset_index(drop=True)
    test_csv_2 = test_csv.iloc[half:].reset_index(drop=True)

    dataset_1 = EvalDataset(test_csv_1, tokenizer)
    dataset_2 = EvalDataset(test_csv_2, tokenizer)

    results = {}

    t0 = Thread(target=run_inference, args=(dataset_1, tokenizer, model_1, 'cuda:0', results, 0))
    t1 = Thread(target=run_inference, args=(dataset_2, tokenizer, model_2, 'cuda:1', results, 1))

    t0.start()
    t1.start()

    t0.join()
    t1.join()

    predictions = results.get(0, []) + results.get(1, [])

    submission_df = pd.DataFrame(predictions, columns=['winner_model_a', 'winner_model_b', 'winner_tie'])
    submission_df['id'] = test_csv['id']
    submission_df.to_csv("submission_llama.csv", index=False)
    del model_1, model_2, tokenizer, dataset_1, dataset_2, results, submission_df, test_csv, test_csv_1, test_csv_2
    torch.cuda.empty_cache()
    gc.collect()


cfg = {
    "model_path": "/kaggle/input/llama-3/transformers/8b-chat-hf/1",
    "adapter_path": "/kaggle/input/meta-llama-3-8b-instruct-2-epoch-label-smoothing/Meta-Llama-3-8B-Instruct-3096-2-epoch-label-smoothing",
    "max_len": 2048,
    "test_csv": "/kaggle/input/lmsys-chatbot-arena/test.csv",
    "batch_size": 1,
}

if __name__ == "__main__":
    main(cfg)

import pandas as pd

llama_preds = pd.read_csv("submission_llama.csv")
gemma_preds = pd.read_csv("submission_gemma.csv")
llama_preds['id'] = llama_preds['id'].astype(int)
gemma_preds['id'] = gemma_preds['id'].astype(int)
llama_preds = llama_preds.sort_values('id').reset_index(drop=True)
gemma_preds = gemma_preds.sort_values('id').reset_index(drop=True)
preds_values = llama_preds[['winner_model_a', 'winner_model_b', 'winner_tie']].values * 0.35 + gemma_preds[
    ['winner_model_a', 'winner_model_b', 'winner_tie']].values * 0.65

submission = pd.DataFrame(preds_values, columns=['winner_model_a', 'winner_model_b', 'winner_tie'])
submission['id'] = llama_preds['id']
submission.to_csv("submission.csv", index=False)
