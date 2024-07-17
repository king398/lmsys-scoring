import numpy as np
import pandas as pd
import torch
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
from peft import PeftModel
from tqdm import tqdm


def tokenize_batch(tokenizer, prompts, responses_a, responses_b, max_length=1536):
    prompts = ["<prompt>: " + p for p in prompts]
    responses_a = ["\n\n<response_a>: " + r_a for r_a in responses_a]
    responses_b = ["\n\n<response_b>: " + r_b for r_b in responses_b]

    texts = [p + r_a + r_b for p, r_a, r_b in zip(prompts, responses_a, responses_b)]
    tokenized = tokenizer(texts, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
    return tokenized.input_ids, tokenized.attention_mask


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def process_batch(batch, model, tokenizer, device):
    input_ids, attention_mask = tokenize_batch(tokenizer, batch['prompt'], batch['response_a'], batch['response_b'],
                                               max_length=1024)
    inputs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}

    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model(**inputs, output_hidden_states=True)

    embeddings = mean_pooling(outputs['hidden_states'][-1], attention_mask.to(device))
    return embeddings.cpu()


# Load data and model
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/lmsys-33k-deduplicated.csv", encoding='utf-8')
df_model = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", encoding='utf-8')
df = df[df['model_a'].isin(df_model['model_a']) & df['model_b'].isin(df_model['model_b'])].reset_index(drop=True)
tokenizer = GemmaTokenizerFast.from_pretrained("google/gemma-2-9b-it", trust_remote_code=True)
model = Gemma2ForSequenceClassification.from_pretrained("google/gemma-2-9b-it", device_map="cuda:1",
                                                        trust_remote_code=True, torch_dtype=torch.float16, num_labels=3,
                                                        )
model = PeftModel.from_pretrained(model,
                                  "/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-bnb-4bit_fold0_1024_full/gemma-2-9b-it-bnb-4bit_fold0_1024_full")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Process data in batches
batch_size = 1  # Adjust based on your GPU memory
embeddings_all = []

for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i + batch_size]
    embeddings = process_batch(batch, model, tokenizer, device)
    embeddings_all.append(embeddings)

embeddings_all = torch.cat(embeddings_all, 0)
embeddings_all = embeddings_all.numpy()

# Save results
np.save("/home/mithil/PycharmProjects/lmsys-scoring/data/embeddings/gemma-2-9b-it-bnb-4bit_fold0_1024_full_valid.npy",
        embeddings_all)
np.save("/home/mithil/PycharmProjects/lmsys-scoring/data/labels_valid.npy", df['label'].values)
