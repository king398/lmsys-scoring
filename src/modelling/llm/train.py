import gc
import os
import warnings

import pandas as pd
import torch.nn
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, \
    BitsAndBytesConfig, DataCollatorWithPadding, Trainer
from utils import seed_everything, find_all_linear_names, compute_metrics
from model import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CFG = {
    'seed': 42,
    'train_csv': '/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama_extra.csv',
    'model_name': 'google/gemma-2-9b-it',
    'max_len': 2560,
    'batch_size': 1,
    'num_classes': 3,
    'model_dir': '/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-2560-extra-data-kl-divergence',
    'epochs': 2,
    'lr': 2e-4,
    'mixed_precision': "bf16",
}
os.environ['WANDB_PROJECT'] = 'lmsys-winner'


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("targets").long()
        logits_labels = inputs.pop("logits")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits_labels, dim=-1), reduction='batchmean')
        loss = loss * 0.9 + kl_loss * 0.1
        # ArcFace loss

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
    valid_df = df[(df['fold'] == fold)].reset_index(drop=True)
    valid_df['len'] = valid_df['text'].apply(lambda x: len(tokenizer(x)['input_ids']))
    train_dataset = Dataset.from_dict(
        {"text": train_df['text'], "targets": train_df['label'], 'logits': train_df[['A_log', 'B_log', 'tie_log']].values})
    valid_dataset = Dataset.from_dict(
        {"text": valid_df['text'], "targets": valid_df['label'], 'logits': valid_df[['A_log', 'B_log', 'tie_log']].values})
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
        eval_strategy="epoch",
        metric_for_best_model="log_loss",
        label_names=["targets"],
        per_device_eval_batch_size=1,
        save_steps=500

    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, quantization_config=quant_config,
                                                 attn_implementation="eager")
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
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    for name, param in model.linear_head.named_parameters():
        param.requires_grad = True
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, cfg['max_len']), num_proc=16)
    valid_dataset = valid_dataset.map(lambda x: tokenize_function(x, tokenizer, cfg['max_len']), num_proc=16)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_dataset = train_dataset.remove_columns(["text"])
    valid_dataset = valid_dataset.remove_columns(["text"])
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        eval_dataset=valid_dataset,

    )

    trainer.train()
    trainer.save_model(cfg['model_dir'])


if __name__ == '__main__':
    main(CFG)
