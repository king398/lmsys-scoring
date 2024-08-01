import os
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    LlamaForSequenceClassification,
    LlamaTokenizerFast,
    PreTrainedTokenizerBase,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score
import random
# import groupkfold
from sklearn.model_selection import StratifiedKFold

os.environ['WANDB_PROJECT'] = 'lmsys-winner'


@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # 4-bit quantized gemma-2-9b-instruct
    max_length: int = 4096
    n_splits: int = 5
    fold_idx: int = 1
    optim_type: str = "adamw_torch"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # global batch size is 8
    per_device_eval_batch_size: int = 1
    n_epochs: int = 2
    freeze_layers: int = 16  # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 4e-5
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = "none"


config = Config()

training_args = TrainingArguments(
    output_dir=f"Meta-Llama-3.1-8B-Instruct-bnb-4bit_fold{config.fold_idx}_{config.max_length}_lowerlr_quantize",
    overwrite_output_dir=True,
    report_to="wandb",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=200,
    optim=config.optim_type,
    fp16=True,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,
)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'linear_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('linear_head')
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    if 'linear_head_2' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('linear_head_2')

    return list(lora_module_names)


tokenizer = LlamaTokenizerFast.from_pretrained(config.checkpoint)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=False,
                                  )
model = LlamaForSequenceClassification.from_pretrained(
    config.checkpoint,
    num_labels=3,
    torch_dtype=torch.float16,
    # device_map="auto",
    attn_implementation="flash_attention_2",
    quantization_config=quant_config,
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules=find_all_linear_names(model),
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["linear_head", ],
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model

model.print_trainable_parameters()
df = pd.read_csv("data/train_folds_llama.csv")
df_add = pd.read_csv("data/lmsys-33k-deduplicated.csv")
df['id'] = df['id'].astype('str')

sgkf = StratifiedKFold(n_splits=5)
for fold, (_, val) in enumerate(sgkf.split(df, df['label'])):
    df.loc[val, "fold"] = int(fold)

# Concatenate dataframes
train_df = df[df['fold'] != config.fold_idx].reset_index(drop=True)
common_columns = df.columns.intersection(df_add.columns)
train_df = train_df[common_columns]
df_add = df_add[common_columns]
train_df = pd.concat([train_df, df_add], axis=0).reset_index(drop=True)
print(train_df)
valid_df = df[df['fold'] == config.fold_idx].reset_index(drop=True)
valid_df = valid_df[common_columns]
train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)


class CustomTokenizer:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        if random.random() > 0.5:
            prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
            response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
            response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
            texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
            tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True, padding="do_not_pad")
            labels = []
            for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
                if a_win:
                    label = 0
                elif b_win:
                    label = 1
                else:
                    label = 2
                labels.append(label)
            return {**tokenized, "labels": labels}
        else:
            # i want to swap response_a and response_b and corresponding labels
            prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
            response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_b"]]
            response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_a"]]
            texts = [p + r_b + r_a for p, r_a, r_b in zip(prompt, response_a, response_b)]
            tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True, padding="do_not_pad")
            labels = []
            for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
                if a_win:
                    label = 1
                elif b_win:
                    label = 0
                else:
                    label = 2
                labels.append(label)
            return {**tokenized, "labels": labels}

    @staticmethod
    def process_text(text: str) -> str:
        return " ".join(eval(text, {"null": ""}))


encode = CustomTokenizer(tokenizer, max_length=config.max_length)
train_ds = train_ds.map(encode, batched=True)
valid_ds = valid_ds.map(encode, batched=True)


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}


trainer = Trainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
)
trainer.train()
