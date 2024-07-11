import torch
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding, \
    LlamaPreTrainedModel
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from threading import Thread, Lock
import ast
from torch import nn

torch.backends.cuda.enable_mem_efficient_sdp(True)
lock = Lock()


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []


def prepare_input(row, tokenizer):
    text = f""""""

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"Instruction:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}\n\n"
    messages = [
        {"role": "user", "content": text},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    inputs = tokenizer.encode_plus(text, return_tensors=None, truncation=True, max_length=cfg['max_len'])

    return inputs['input_ids'], inputs['attention_mask'], len(inputs['input_ids'])


class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.input_ids = df['input_ids']
        self.attention_mask = df['attention_mask']
        self.id = df['id']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if idx >= len(self.input_ids):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self.input_ids)}")
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'id': self.id[idx]}


def run_inference(dataset, tokenizer, model, device, results, index):
    predictions = []
    ids = []
    for batch in tqdm(
            DataLoader(dataset, batch_size=cfg['batch_size'], collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
                       pin_memory=True, prefetch_factor=8, num_workers=4)):
        ids.extend(batch.pop('id'))
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad() and torch.cuda.amp.autocast():
            outputs = model(batch)
        batch_predictions = outputs['logits'].softmax(dim=-1).detach().cpu().numpy().tolist()

        predictions.extend(batch_predictions)
        del batch, outputs
        gc.collect()

    with lock:
        update_data = []
        for i, pred in zip(ids, predictions):
            update_data.append(
                {'winner_model_a': pred[0], 'winner_model_b': pred[1], 'winner_tie': pred[2], 'id': str(i)})

        update_df = pd.DataFrame(update_data)
        results[index] = update_df


class GemmaClassifier(LlamaPreTrainedModel):
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
        hidden_states = self.mean_pooling(hidden_states, tensors['attention_mask']).type(torch.bfloat16)

        return {"logits": self.linear_head(hidden_states)}


def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], trust_remote_code=True)
    test_csv = pd.read_csv(cfg['test_csv'], encoding='utf-8')
    test_csv['prompt'] = test_csv['prompt'].apply(string_to_list)
    test_csv['response_a'] = test_csv['response_a'].apply(string_to_list)
    test_csv['response_b'] = test_csv['response_b'].apply(string_to_list)

    # Apply the prepare_input function and add new columns directly
    test_csv['input_ids'], test_csv['attention_mask'], test_csv['input_length'] = zip(
        *test_csv.apply(prepare_input, axis=1, args=(tokenizer,)))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False)

    model_1 = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.bfloat16,
                                                   device_map="cuda:0", trust_remote_code=True,
                                                   quantization_config=bnb_config, attn_implementation="eager", )
    model_1 = GemmaClassifier(model_1, "cuda:0").to("cuda:0")
    model_1.load_adapter(cfg['adapter_path'])

    model_2 = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.bfloat16,
                                                   device_map="cuda:1", trust_remote_code=True,
                                                   quantization_config=bnb_config, attn_implementation="eager", )
    model_2 = GemmaClassifier(model_2, "cuda:1").to("cuda:1")
    model_2.load_adapter(cfg['adapter_path'])

    tokenizer.pad_token = tokenizer.eos_token

    half = len(test_csv) // 2
    test_csv_1 = test_csv.iloc[:half].reset_index(drop=True)
    test_csv_2 = test_csv.iloc[half:].reset_index(drop=True)
    # short by input length
    test_csv_1 = test_csv_1.sort_values('input_length', ascending=False).reset_index(drop=True)
    test_csv_2 = test_csv_2.sort_values('input_length', ascending=False).reset_index(drop=True)
    dataset_1 = EvalDataset(test_csv_1, tokenizer)
    dataset_2 = EvalDataset(test_csv_2, tokenizer)
    results = {}
    t0 = Thread(target=run_inference, args=(dataset_1, tokenizer, model_1, 'cuda:0', results, 0))
    t1 = Thread(target=run_inference, args=(dataset_2, tokenizer, model_2, 'cuda:1', results, 1))

    t0.start()
    t1.start()

    t0.join()
    t1.join()
    submission_df = pd.concat([results[0], results[1]], ignore_index=True)
    submission_df.to_csv("submission.csv", index=False)


cfg = {
    "model_path": "/kaggle/input/llama-3/transformers/8b-chat-hf/1",
    "adapter_path": "/kaggle/input/meta-llama-3-8b-instruct-0-2-smoothing/Meta-Llama-3-8B-Instruct-0-2-smoothing",
    "max_len": 2048,
    "test_csv": "/kaggle/input/lmsys-chatbot-arena/test.csv",
    "batch_size": 4,
}

if __name__ == "__main__":
    main(cfg)
