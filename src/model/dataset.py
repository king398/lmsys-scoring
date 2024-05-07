from torch.utils.data import Dataset
import torch
import pandas as pd
import transformers


class WinnerDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = df['text']
        self.label = df['label']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = str(self.text[item])
        score = self.label[item]
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
            "targets": torch.tensor(score, dtype=torch.long)

        }


