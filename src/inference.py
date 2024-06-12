import torch
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding, LlamaPreTrainedModel
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
    def __init__(self, model, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3).cuda()

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
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False)

    model_1 = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.float16,
                                                   device_map="cuda:0", trust_remote_code=True,
                                                   quantization_config=bnb_config)
    model_1 = LLamaClassifier(model_1)
    # model_1.gradient_checkpointing_enable()
    model_2 = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.float16,
                                                   device_map="cuda:1", trust_remote_code=True,
                                                   quantization_config=bnb_config)
    model_2 = LLamaClassifier(model_2)
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
    submission_df.to_csv("submission.csv", index=False)


cfg = {
    "model_path": "/kaggle/input/meta-llama-3-8b-instruct-2560-2-epoch-all-logits-m/Meta-Llama-3-8B-Instruct-2560-2-epoch-all-logits/",
    "max_len": 3096,
    "test_csv": "/kaggle/input/lmsys-chatbot-arena/test.csv",
    "batch_size": 1,
}

if __name__ == "__main__":
    main(cfg)
