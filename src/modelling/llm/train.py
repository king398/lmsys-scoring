import gc
import os
import warnings

import numpy as np
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
    'train_csv': '/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv',
    'model_name': 'unsloth/Meta-Llama-3.1-8B-bnb-4bit',
    'max_len': 3096,
    'batch_size': 1,
    'num_classes': 3,
    'model_dir': '/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-bnb-4bit-training',
    'epochs': 2,
    'lr': 4e-5,
    'mixed_precision': "bf16",
    'gradient_accumulation_steps': 16,
    "embedding_path": "/home/mithil/PycharmProjects/lmsys-scoring/data/embeddings/hidden_states.npy"
}
os.environ['WANDB_PROJECT'] = 'lmsys-winner'


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("targets").long()
        embeddings = inputs.pop("embeddings")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        hidden_states = outputs.get('features')
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        # ArcFace loss
        cosine_loss = F.cosine_embedding_loss(hidden_states, embeddings, torch.ones(logits.shape[0]).to(logits.device))
        loss = loss + cosine_loss * 0.1
        return (loss, outputs) if return_outputs else loss


def tokenize_function(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="longest", max_length=max_length)
    return tokenized_inputs


def main(cfg):
    warnings.filterwarnings("ignore")
    seed_everything(cfg['seed'])

    gc.enable()
    df = pd.read_csv(cfg['train_csv'])
    embeddings = np.load(cfg['embedding_path'])
    df['embeddings'] = embeddings.tolist()
    fold = 0
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], add_eos_token=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    train_df['len'] = train_df['text'].apply(lambda x: len(tokenizer(x)['input_ids']))
    valid_df = df[(df['fold'] == fold)].reset_index(drop=True)
    valid_df['len'] = valid_df['text'].apply(lambda x: len(tokenizer(x)['input_ids']))
    train_dataset = Dataset.from_dict(
        {"text": train_df['text'], "targets": train_df['label'], "embeddings": train_df['embeddings']})
    valid_dataset = Dataset.from_dict(
        {"text": valid_df['text'], "targets": valid_df['label'], "embeddings": valid_df['embeddings']})
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg['batch_size'],
        num_train_epochs=cfg['epochs'],
        bf16=True,
        output_dir=cfg['model_dir'],
        gradient_checkpointing=True,
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        save_strategy="steps",
        overwrite_output_dir=True,
        learning_rate=float(cfg['lr']),
        optim="adamw_8bit",
        seed=42,
        logging_steps=1,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        save_safetensors=True,
        run_name=cfg['model_dir'].split("/")[-1],
        remove_unused_columns=False,
        eval_strategy="steps",
        metric_for_best_model="log_loss",
        label_names=["targets"],
        per_device_eval_batch_size=1,
        save_steps=300,
        eval_steps=300,
        load_best_model_at_end=True,

    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=quant_config,
                                                 attn_implementation="flash_attention_2", )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("Linear layers: ", find_all_linear_names(model))
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=find_all_linear_names(model),
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["linear_head_1", "linear_head_2"],
    )
    model = LLamaClassifier(model, torch.bfloat16)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
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
