import torch
import sklearn
import numpy as np
import pandas as pd
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import gc

test = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/train.csv')


def process_text(text: str) -> str:
    return " ".join(eval(text, {"null": ""}))


test.loc[:, 'prompt'] = test['prompt'].apply(process_text)
test.loc[:, 'response_a'] = test['response_a'].apply(process_text)
test.loc[:, 'response_b'] = test['response_b'].apply(process_text)


class Config:
    gemma_dir = '/home/mithil/PycharmProjects/lmsys-scoring/models/kaggler_model/checkpoint-4916'
    max_length = 2560
    spread_max_length = False
    batch_size = 1


cfg = Config()


def tokenize(
        tokenizer, prompt, response_a, response_b, max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]
    if spread_max_length:
        prompt = tokenizer(prompt, max_length=max_length // 3, truncation=True, padding=False).input_ids
        response_a = tokenizer(response_a, max_length=max_length // 3, truncation=True, padding=False).input_ids
        response_b = tokenizer(response_b, max_length=max_length // 3, truncation=True, padding=False).input_ids
        input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        attention_mask = [[1] * len(i) for i in input_ids]
    else:
        text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = tokenizer(text, max_length=max_length, truncation=True, padding="do_not_pad")
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
    return input_ids, attention_mask


tokenizer = GemmaTokenizerFast.from_pretrained("google/gemma-2-9b-it")
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

data = pd.DataFrame()
data["id"] = test["id"]
data["input_ids"], data["attention_mask"] = tokenize(tokenizer, test["prompt"], test["response_a"], test["response_b"])
data["length"] = data["input_ids"].apply(len)
gc.collect()
model_0 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map="cuda:1",
    use_cache=False,
    torch_dtype=torch.float16
)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

from tqdm import tqdm
@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    hidden_states_all = None

    ids = df["id"].to_list()

    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device), output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        hidden_states = mean_pooling(hidden_states, inputs["attention_mask"]).type(torch.float16)
        if hidden_states_all is None:
            hidden_states_all = hidden_states
        else:
            hidden_states_all = torch.cat([hidden_states_all, hidden_states], dim=0)

    return hidden_states_all.cpu()


hidden_states = inference(data, model_0, torch.device("cuda:1"))

hidden_states = hidden_states.numpy()
np.save("data/embeddings/hidden_states.npy", hidden_states)
