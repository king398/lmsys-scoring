import gc
import os
import warnings

import pandas as pd
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, TrainingArguments, MistralForSequenceClassification, \
    BitsAndBytesConfig, DataCollatorWithPadding, Trainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils import seed_everything, find_all_linear_names, compute_metrics
import torch
cfg = {
    'seed': 42,
    'train_csv': '/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv',
    'model_name': 'prometheus-eval/prometheus-7b-v2.0',
    'max_len': 2560,
    'batch_size': 1,
    'num_classes': 3,
    'model_dir': '/home/mithil/PycharmProjects/lmsys-scoring/models/Promethus-eval-2560-2-epoch-classification',
    'epochs': 2,
    'lr': 4e-5,
    'mixed_precision': "bf16",
}

#os.environ['WANDB_PROJECT'] = 'lmsys-winner'


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
    #train_df['len'] = train_df['text'].apply(lambda x: len(tokenizer(x)['input_ids']))
    # train_df = train_df[train_df['len'] < cfg['max_len']].reset_index(drop=True)

    train_dataset = Dataset.from_dict({"text": train_df['text'], 'labels': train_df['label']})
    train_dataset = train_dataset.map(tokenize_function, batched=False,
                                      fn_kwargs={"tokenizer": tokenizer, "max_length": cfg['max_len']}, num_proc=16)

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
        report_to="none"

    )

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = MistralForSequenceClassification.from_pretrained(cfg['model_name'], trust_remote_code=True,
                                                             attn_implementation="flash_attention_2",
                                                             torch_dtype=torch.float16,
                                                             quantization_config=quant_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=find_all_linear_names(model),
        task_type="SEQ_CLS",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    response_template = "[RESULT]:"

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,max_length=cfg['max_len'] )

    trainer = Trainer(
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
    main(cfg)
