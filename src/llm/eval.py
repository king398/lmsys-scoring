from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss
import ast
from torch.utils.data import DataLoader, Dataset
import math

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-7b-v2.0", torch_dtype=torch.float16,
                                             device_map="auto",
                                             trust_remote_code=True, attn_implementation="flash_attention_2", )
model.load_adapter("/home/mithil/PycharmProjects/lmsys-scoring/models/Prometheus-eval-2-epoch-2560-len")
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


# Define the Dataset
class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.text = df['text']
        self.tokenizer = tokenizer
        self.label = df['label']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {"text": self.text[idx], "label": self.label[idx]}


# Prepare the dataset
df['text'] = df.apply(prepare_input, axis=1)
dataset = EvalDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)

predictions = []
labels = []

# Evaluate the model
for batch in tqdm(dataloader):
    batch['text'] += f"[RESULT]:"
    inputs = tokenizer(batch['text'], return_tensors="pt", truncation=True, max_length=3096, padding="longest")
    for k, v in inputs.items():
        inputs[k] = v.to("cuda:1")

    outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, output_scores=True,
                             return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
    scores = outputs['scores'][0].softmax(dim=1)

    # Extract predictions for specific tokens
    target_token_ids = [330, 365, 14628]  # Token IDs for "A", "B", "tie"
    batch_predictions = scores[:, target_token_ids].detach().cpu().numpy().tolist()
    predictions.extend(batch_predictions)
    labels.append(batch['label'].tolist())

# Calculate and print log loss
print(log_loss(labels, predictions))
# save to a oof file
oof_df = pd.DataFrame(predictions, columns=["A", "B", "tie"])
oof_df["label"] = df["label"]
oof_df['id'] = df['id']
# log loss per row
log_losses = []

for i in range(len(df)):
    log_loss_value = log_loss([labels[i]], [predictions[i]], labels=[0, 1, 2])
    log_losses.append(log_loss_value)

oof_df["log_loss"] = log_losses

oof_df['perplexity'] = oof_df.apply(lambda x: math.e ** x["log_loss"], axis=1)
oof_df.to_csv("/home/mithil/PycharmProjects/lmsys-scoring/models/Prometheus-eval-2-epoch-2560-len/oof.csv", index=False)
