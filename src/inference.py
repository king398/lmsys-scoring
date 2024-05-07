import gc

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoConfig
import torch
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast
import random
import os
import numpy as np
from typing import List
from torch import nn
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed=42):
    """Seed everything to ensure reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class WinnerDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = df['text']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = str(self.text[item])
        encoding = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            max_length=self.max_len,
            truncation=True,
        )
        for k, v in encoding.items():
            encoding[k] = torch.tensor(v, dtype=torch.long)
        return {
            **encoding,

        }


def inference_loop(
        inference_dataloader,
        model,
        device,

) -> torch.Tensor:
    model.eval()
    predictions = []
    for batch in tqdm(inference_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad() and autocast():

            output = model(batch).softmax(dim=1).detach().cpu()
            predictions.append(output)
    predictions = torch.cat(predictions, dim=0)
    return predictions


cfg = {
    "seed": 42,
    "test_csv": "/kaggle/input/lmsys-chatbot-arena/test.csv",
    "model_dir": "/kaggle/input/lmsys-models/deberta-v3-xsmall-baseline",
    "base_model": "/kaggle/input/deberta-v3-xsmall-offline/deberta-v3-xsmall",
    "max_len": 1536,
    "batch_size": 6,
    "folds": [
        0, 1, 2, 3,
    ]

}


def create_text(row):
    text = ""
    for prompt, response_a, response_b in zip(row['prompt'], row['response_a'], row['response_b']):
        text += f"Prompt: {prompt} Response_A: {response_a} Response_B: {response_b} "
    return text


def prepare_df(df):
    df['text'] = df.apply(create_text, axis=1)
    return df


class WinnerModel(nn.Module):
    def __init__(self, model_name, num_labels, tokenizer):
        super(WinnerModel, self).__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, inputs):
        return self.model(**inputs).logits


def main(cfg):
    seed_everything(cfg['seed'])
    df = pd.read_csv(cfg['test_csv'])
    df = prepare_df(df)
    gc.enable()

    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'])
    test_dataset = WinnerDataset(df=df, max_len=cfg['max_len'], tokenizer=tokenizer)
    datacollator = DataCollatorWithPadding(tokenizer)
    inference_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False,
                                                       collate_fn=datacollator,pin_memory=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_array = None
    for fold in cfg['folds']:
        model = WinnerModel(cfg['base_model'], 3, tokenizer)
        model.load_state_dict(torch.load(f"{cfg['model_dir']}/fold_{fold}.pth"))
        model.to(device)
        model = nn.DataParallel(model)
        predictions = inference_loop(inference_dataloader, model, device)
        if predictions_array is None:
            predictions_array = predictions
        else:
            predictions_array += predictions
        torch.cuda.empty_cache()
        del model
    predictions_array /= len(cfg['folds'])
    predictions_array = predictions_array.numpy()

    prediction_df = pd.DataFrame({'id': df['id']})
    prediction_df['winner_model_a'] = predictions_array[:, 0]
    prediction_df['winner_model_b'] = predictions_array[:, 1]
    prediction_df['winner_tie'] = predictions_array[:, 2]
    prediction_df.to_csv("submission.csv", index=False)
    print(prediction_df.head())


main(cfg)
