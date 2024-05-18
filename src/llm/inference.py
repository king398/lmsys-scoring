from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import torch
from tqdm import tqdm
import ast
from torch.utils.data import DataLoader, Dataset
import gc
from peft import PeftModel

torch.backends.cuda.enable_mem_efficient_sdp(False)


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []  # Handle cases where conversion fails


def prepare_input(row, tokenizer):
    text = """Please analyze the conversation below between a human and two language models which give both respectively give the response ###Response A and ###Response B. The models are each asked to respond to the same prompts which is indicated by ###Instruction:. 
After reviewing the responses from both models, please determine which is the better response overall - Response_a, Response_b, or was it a tie? Respond with only a single word after [RESULT]: . Either "A" if ###Response A was better, "B" if ###Response B was better, or "tie" if their responses were equally good or bad"""

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"""
###Instruction:: {prompt} 
###Response A: {response_a} 
###Response B: {response_b}"""
    text = tokenizer.decode(
        tokenizer(text, return_tensors="pt", truncation=True, max_length=2000, padding=True)['input_ids'][0])

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
        return {"text": self.text[idx]}


def main(cfg):
    model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.float16,
                                                 device_map=cfg['device_map'], trust_remote_code=True)

    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(model, cfg['adapter_path'])
    model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], trust_remote_code=True)
    test_csv = pd.read_csv(cfg['test_csv'], encoding='utf-8')[:100]
    test_csv['prompt'] = test_csv['prompt'].apply(string_to_list)
    test_csv['response_a'] = test_csv['response_a'].apply(string_to_list)
    test_csv['response_b'] = test_csv['response_b'].apply(string_to_list)
    test_csv['text'] = test_csv.apply(prepare_input, axis=1, args=(tokenizer,))
    tokenizer.pad_token = tokenizer.eos_token

    predictions = []
    dataset = EvalDataset(test_csv, tokenizer)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4, pin_memory=True)

    for batch in tqdm(dataloader):
        batch['text'] += "[RESULT]:"
        inputs = tokenizer(batch['text'], return_tensors="pt", truncation=True, max_length=cfg['max_len'], padding=True)
        for k, v in inputs.items():
            inputs[k] = v.cuda()

        with torch.no_grad() and torch.cuda.amp.autocast():
            outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, output_scores=True,
                                     return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
        scores = outputs['scores'][0].softmax(dim=1)

        # Extract predictions for specific tokens
        target_token_ids = [330, 365, 14628]  # Token IDs for "A", "B", "tie"
        batch_predictions = scores[:, target_token_ids].detach().cpu().numpy().tolist()
        predictions.extend(batch_predictions)
        del inputs, outputs
        gc.collect()

    submission_df = pd.DataFrame(predictions, columns=['winner_model_a', 'winner_model_b', 'winner_tie'])
    submission_df['id'] = test_csv['id']
    submission_df.to_csv("submission.csv", index=False)


cfg = {
    "model_path": "/kaggle/input/promethus-eval-model",
    "adapter_path": "/kaggle/input/prometheus-eval",
    "max_len": 2048,
    "tokens": [330, 365, 14628],
    "test_csv": "/kaggle/input/lmsys-chatbot-arena/train.csv",
    "batch_size": 2,
    'device_map': {
        'model.embed_tokens': 0,
        'model.layers.0': 0,
        'model.layers.1': 0,
        'model.layers.2': 0,
        'model.layers.3': 0,
        'model.layers.4': 0,
        'model.layers.5': 0,
        'model.layers.6': 0,
        'model.layers.7': 0,
        'model.layers.8': 0,
        'model.layers.9': 0,
        'model.layers.10': 0,
        'model.layers.11': 0,
        'model.layers.12': 0,
        'model.layers.13': 0,
        'model.layers.14': 0,
        'model.layers.15': 0,
        'model.layers.16': 0,
        'model.layers.17': 0,
        'model.layers.18': 0,
        'model.layers.19': 0,
        'model.layers.20': 0,
        'model.layers.21': 0,
        'model.layers.22': 0,
        'model.layers.23': 0,
        'model.layers.24': 0,
        'model.layers.25': 0,
        'model.layers.26': 0,
        'model.layers.27': 1,
        'model.layers.28': 1,
        'model.layers.29': 1,
        'model.layers.30': 1,
        'model.layers.31': 1,

        'model.norm': 1,
        'lm_head': 1
    }
}

if __name__ == "__main__":
    main(cfg)
