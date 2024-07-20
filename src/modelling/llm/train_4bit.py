import os
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score
import random
from transformers import Trainer, TrainingArguments, get_scheduler
# import groupkfold
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW

import math
import torch
from transformers.optimization import AdamW
from transformers.optimization import Adafactor, get_scheduler, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim import lr_scheduler
import bitsandbytes as bnb


def create_scheduler(config, model, num_training_steps: int, optimizer: torch.optim.Optimizer, lr_scheduler_type):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.
    Args:
      num_training_steps (int): The number of training steps to do.
    """

    if lr_scheduler_type == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0)
    elif lr_scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, eta_min=0)
    else:
        scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    return scheduler


def create_optimizer(args, model):
    """
    Setup the optimizer.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.

    MODIFIED VERSION:
    * added support for differential learning rates per layer

    reference: https://github.com/huggingface/transformers/blob/05fa1a7ac17bb7aa07b9e0c1e138ecb31a28bbfe/src/transformers/trainer.py#L804
    """

    no_decay = ["bias", "LlamaRMSNorm.weight", "LlamaRMSNorm.bias"]
    ### ADDED
    print(args.weight_decay)
    if args.discriminative_learning_rate:

        num_layers = model.config.num_hidden_layers
        print("num_layers : ", num_layers)

        learning_rate_powers = range(0, num_layers, num_layers // args.discriminative_learning_rate_num_groups)
        layer_wise_learning_rates = [
            pow(args.discriminative_learning_rate_decay_rate, power) * args.lr
            for power in learning_rate_powers
            for _ in range(num_layers // args.discriminative_learning_rate_num_groups)
        ]
        layer_wise_learning_rates = layer_wise_learning_rates[::-1]
        print('Layer-wise learning rates:', layer_wise_learning_rates)

        # group embedding paramters from the transformer encoder
        embedding_layer = model.base_model.model.model.embed_tokens
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in embedding_layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": pow(args.discriminative_learning_rate_decay_rate, num_layers) * args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in embedding_layer.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": pow(args.discriminative_learning_rate_decay_rate, num_layers) * args.lr,
                "weight_decay": 0.0,
            },
        ]

        # group encoding paramters from the transformer encoder
        encoding_layers = [layer for layer in model.base_model.model.model.layers]
        for i, layer in enumerate(encoding_layers):
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "lr": layer_wise_learning_rates[i],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "lr": layer_wise_learning_rates[i],
                    "weight_decay": 0.0,
                },
            ]
            # print(f"Detected unattached modules in model.encoder: {[n for n, p in model.model.layers.named_parameters() if not n.startswith('layer')]}")
        # optimizer_grouped_parameters += [
        #     {
        #         "params": [p for n, p in model.backbone.encoder.named_parameters() if not n.startswith('layer') and not any(nd in n for nd in no_decay)],
        #         "lr": layer_wise_learning_rates[-1],
        #         "weight_decay": args.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.backbone.encoder.named_parameters() if not n.startswith('layer') and any(nd in n for nd in no_decay)],
        #         "lr": layer_wise_learning_rates[-1],
        #         "weight_decay": 0.0,
        #     },
        # ]

        # group paramters from the task specific head
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.named_parameters() if
                           'score' in n and not any(nd in n for nd in no_decay)],
                "lr": args.head_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if 'score' in n and any(nd in n for nd in no_decay)],
                "lr": args.head_lr,
                "weight_decay": 0.0,
            },
        ]
    ### END ADDED
    else:
        # group paramters for the entire network
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": args.learning_rate,
                "weight_decay": 0.0,
            },
        ]

    if args.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}

    elif args.optim_type == 'paged_adamw_8bit':
        optimizer_cls = bnb.optim.PagedAdamW8bit
        optimizer_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.eps,
        }
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.eps,
        }
    # print(optimizer_grouped_parameters)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    # make sure to optimize nn.Embedding with 32-bit AdamW
    # reference: https://github.com/huggingface/transformers/issues/14819#issuecomment-1003445038
    if optimizer_cls.__name__ == "Adam8bit":
        manager = bnb.optim.GlobalOptimManager.get_instance()
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                manager.register_module_override(module, "weight", {"optim_bits": 32})
    # print(optimizer_grouped_parameters)
    return optimizer


os.environ['WANDB_PROJECT'] = 'lmsys-winner'


@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "unsloth/gemma-2-9b-bnb-4bit"  # 4-bit quantized gemma-2-9b-instruct
    max_length: int = 2560
    n_splits: int = 5
    fold_idx: int = 0
    # optim_type: str = "adamw_torch"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2  # global batch size is 8
    per_device_eval_batch_size: int = 8
    n_epochs: int = 2
    freeze_layers: int = 16  # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 6e-5
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = "none"


config = Config()


class CustomTrainingArguments(TrainingArguments):
    def __init__(self, discriminative_learning_rate=False, discriminative_learning_rate_num_groups=1,
                 discriminative_learning_rate_decay_rate=1.0, lr=1e-6, eps=1e-8, head_lr=None, adafactor=None,
                 optim_type=None, **kwargs):
        super().__init__(**kwargs)
        self.discriminative_learning_rate = discriminative_learning_rate
        self.discriminative_learning_rate_num_groups = discriminative_learning_rate_num_groups
        self.discriminative_learning_rate_decay_rate = discriminative_learning_rate_decay_rate
        self.head_lr = head_lr
        self.lr = lr
        self.adafactor = adafactor
        self.optim_type = optim_type
        self.eps = eps


training_args = CustomTrainingArguments(
    output_dir=f"/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-bnb-4bit_fold{config.fold_idx}_{config.max_length}_lowerlr_noquantize_fullfinetune",
    overwrite_output_dir=True,
    gradient_checkpointing=True,
    report_to="wandb",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=200,
    # optim=config.optim_type,
    bf16=True,
    learning_rate=config.lr,
    lr=config.lr,
    warmup_steps=config.warmup_steps,
    discriminative_learning_rate=True,  # Custom argument
    discriminative_learning_rate_num_groups=6,  # Custom argument, set accordingly
    discriminative_learning_rate_decay_rate=0.9,  # Custom argument, set accordingly
    head_lr=2e-4  # Custom argument

)

lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    # only target self-attention
    target_modules="all-linear",
    # layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.SEQ_CLS,
)

tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
tokenizer.add_eos_token = True  # We'll add <eos> at the end
tokenizer.padding_side = "right"

model = Gemma2ForSequenceClassification.from_pretrained(
    config.checkpoint,
    num_labels=3,
    torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    # device_map="auto",
)
model.config.use_cache = False

model = get_peft_model(model, lora_config)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
config.inference_mode = False
df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv")
df_add = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/lmsys-33k-deduplicated.csv")
df['id'] = df['id'].astype('str')

sgkf = GroupKFold(n_splits=config.n_splits)
for fold, (_, val) in enumerate(sgkf.split(df, df.prompt, df.prompt)):
    df.loc[val, "fold"] = int(fold)

# Concatenate dataframes
train_df = df[df['fold'] != config.fold_idx].reset_index(drop=True)
common_columns = df.columns.intersection(df_add.columns)
train_df = train_df[common_columns]
# df_add = df_add[common_columns]
# train_df = pd.concat([train_df, df_add], axis=0).reset_index(drop=True)
valid_df = df[df['fold'] == config.fold_idx].reset_index(drop=True)
valid_df = valid_df[common_columns]
train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)

optimizer = create_optimizer(training_args, model)
num_training_steps = len(train_ds) // training_args.per_device_train_batch_size * training_args.num_train_epochs

scheduler = create_scheduler(config, model, num_training_steps, optimizer, lr_scheduler_type="linear")


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


class CustomTokenizerValid:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        # if random.random() >= 0.0:
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

    @staticmethod
    def process_text(text: str) -> str:
        return " ".join(eval(text, {"null": ""}))


encode = CustomTokenizer(tokenizer, max_length=config.max_length)
encode_valid = CustomTokenizerValid(tokenizer, max_length=config.max_length)
train_ds = train_ds.map(encode, batched=True)
valid_ds = valid_ds.map(encode_valid, batched=True)


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}


class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(num_training_steps, self.optimizer)

    def create_optimizer(self):
        return create_optimizer(self.args, self.model)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer):
        return create_scheduler(self.args, self.model, num_training_steps, optimizer, self.args.lr_scheduler_type)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train()
