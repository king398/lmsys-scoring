import gc
import os
import warnings

import pandas as pd
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, \
    BitsAndBytesConfig, DataCollatorWithPadding, Trainer
from utils import seed_everything, find_all_linear_names, compute_metrics
import torch
from torch.nn import functional as F
from model import LLamaClassifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CFG = {
    'seed': 42,
    'train_csv': '/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv',
    'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'max_len':  3096,
    'batch_size': 1,
    'num_classes': 3,
    'model_dir': '/home/mithil/PycharmProjects/lmsys-scoring/models/Meta-Llama-3-8B-Instruct-2-epoch-layer-replication',
    'epochs': 2,
    'lr': 4e-5,
    'mixed_precision': "bf16",
}
os.environ['WANDB_PROJECT'] = 'lmsys-winner'


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("targets").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        return (loss, outputs) if return_outputs else loss


def tokenize_function(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="longest", max_length=max_length)
    return tokenized_inputs


def main(cfg):
    warnings.filterwarnings("ignore")
    seed_everything(cfg['seed'])

    gc.enable()
    df = pd.read_csv(cfg['train_csv'])
    fold = 0
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], add_eos_token=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    train_df['len'] = train_df['text'].apply(lambda x: len(tokenizer(x)['input_ids']))
    train_df = train_df[train_df['len'] < cfg['max_len']].reset_index(drop=True)
    valid_df = df[(df['fold'] == fold)].reset_index(drop=True)
    valid_df['len'] = valid_df['text'].apply(lambda x: len(tokenizer(x)['input_ids']))
    train_dataset = Dataset.from_dict({"text": train_df['text'], "targets": train_df['label']})
    valid_dataset = Dataset.from_dict({"text": valid_df['text'], "targets": valid_df['label']})
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg['batch_size'],
        num_train_epochs=cfg['epochs'],
        bf16=True,
        output_dir=cfg['model_dir'],
        gradient_checkpointing=True,
        gradient_accumulation_steps=16,
        save_strategy="epoch",
        overwrite_output_dir=True,
        learning_rate=float(cfg['lr']),
        optim="adamw_8bit",
        seed=42,
        logging_steps=1,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_safetensors=True,
        run_name=cfg['model_dir'].split("/")[-1],
        remove_unused_columns=False,
        #eval_strategy="epoch",
        #metric_for_best_model="log_loss",
        label_names=["targets"],

    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], trust_remote_code=True,
                                                 attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16, quantization_config=quant_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = LLamaClassifier(model, torch_dtype=torch.bfloat16)
    print("Linear layers: ", find_all_linear_names(model))
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=find_all_linear_names(model),
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["linear_head", ],
        layer_replication=[(0,16),(8,24),(16,32)]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    for name, param in model.linear_head.named_parameters():
        param.requires_grad = True
    print(model)
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, cfg['max_len']))
    valid_dataset = valid_dataset.map(lambda x: tokenize_function(x, tokenizer, cfg['max_len']))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, )
    train_dataset = train_dataset.remove_columns(["text"])
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

    )

    trainer.train()
    trainer.save_model(cfg['model_dir'])


if __name__ == '__main__':
    main(CFG)