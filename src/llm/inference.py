from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import torch
from tqdm import tqdm
import ast
from torch.utils.data import DataLoader, Dataset
import gc
torch.backends.cuda.enable_mem_efficient_sdp(False)


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []  # Handle cases where conversion fails


def main(cfg):
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.float16,
                                                  quantization_config=quant_config,low_cpu_mem_usage=True,
                                                 device_map={'': 0},                   )

    model.load_adapter(cfg['adapter_path'])
    model = torch.compile(model,mode="reduce-overhead",fullgraph=True)
    tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0", trust_remote_code=True)
    test_csv = pd.read_csv(cfg['test_csv'], encoding='utf-8')
    test_csv['prompt'] = test_csv['prompt'].apply(string_to_list)
    test_csv['response_a'] = test_csv['response_a'].apply(string_to_list)
    test_csv['response_b'] = test_csv['response_b'].apply(string_to_list)
    test_csv['text'] = test_csv.apply(prepare_input, axis=1, args=(tokenizer,))
    tokenizer.pad_token = tokenizer.eos_token
    predictions = []
    dataset = EvalDataset(test_csv, tokenizer)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    for batch in tqdm(dataloader):
        batch['text'] += "[RESULT]:"
        inputs = tokenizer(batch['text'], return_tensors="pt", truncation=True, max_length=cfg['max_len'],
                           padding=False)
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, output_scores=True,
                                 return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
        outputs = outputs['scores'][0].softmax(dim=1)
        predictions.extend(outputs[:, [330, 365, 14628]].detach().cpu().numpy().tolist())
        del inputs,outputs
        gc.collect()
    submission_df = pd.DataFrame(predictions, columns=['winner_model_a', 'winner_model_b', 'winner_tie'])
    submission_df['id'] = test_csv['id']
    submission_df.to_csv("submission.csv", index=False)


def prepare_input(row, tokenizer):
    text = """Please analyze the conversation below between a human and two language models which give both respectively give the response ###Response A and ###Response B. The models are each asked to respond to the same prompts which is indicated by ###Instruction:. 
After reviewing the responses from both models, please determine which is the  better responses overall - Response_a, Response_b, or was it a tie? Respond with only a single word after [RESULT]: . Either "A" if ###Response A was better, "B" if ###Response B was better, or "tie" if their responses were equally good or bad"""

    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"""
###Instruction:: {prompt} 
###Response A: {response_a} 
###Response B: {response_b}"""
    messages = [
        {"role": "user", "content": text},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False
                                         )
    return text


class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.text = df['text']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {"text": self.text[idx]}


cfg = {
    "model_path": "/kaggle/input/promethus-eval-model/",
    "adapter_path": "/kaggle/input/prometheus-eval",
    "max_len": 3096,
    "tokens": [330, 365, 14628],
    "test_csv": "/kaggle/input/lmsys-chatbot-arena/train.csv",
    "batch_size": None,

}

if __name__ == "__main__":
    main(cfg)
