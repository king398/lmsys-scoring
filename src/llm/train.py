import gc
import os
import warnings

import pandas as pd
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, AutoModelForCausalLM, \
    BitsAndBytesConfig, Trainer

from utils import seed_everything, find_all_linear_names, compute_metrics
import torch

os.environ['WANDB_PROJECT'] = 'lmsys-winner'


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
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = Dataset.from_dict({"text": train_df['text']})
    train_dataset = train_dataset.map(tokenize_function, batched=False,
                                      fn_kwargs={"tokenizer": tokenizer, "max_length": cfg['max_len']}, num_proc=16)

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg['batch_size'],
        num_train_epochs=cfg['epochs'],
        fp16_full_eval=True,
        fp16=True,
        output_dir=cfg['model_dir'],
        gradient_checkpointing=True,
        gradient_accumulation_steps=16,
        save_strategy="epoch",
        overwrite_output_dir=True,
        learning_rate=float(cfg['lr']),
        optim="adamw_torch",
        seed=42,
        logging_steps=1,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        save_safetensors=True,
    )
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'],
                                                 quantization_config=quant_config,
                                                 )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CASUAL_LM",
        target_modules=find_all_linear_names(model), )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, )
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        trainer.train()
    trainer.save_model(cfg['model_dir'])


cfg = {
    'seed': 42,
    'train_csv': '/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv',
    'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'max_len': 3096,
    'batch_size': 1,
    'num_classes': 3,
    'model_dir': '/home/mithil/PycharmProjects/lmsys-scoring/models/llama-3-8b',
    'epochs': 1,
    'lr': 1e-5,
    'mixed_precision': "fp16",
}

if __name__ == '__main__':
    main(cfg)
