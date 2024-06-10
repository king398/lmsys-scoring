import gc
import os
import warnings

import pandas as pd
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, \
    BitsAndBytesConfig, LlamaPreTrainedModel, DataCollatorWithPadding, Trainer
from utils import seed_everything, find_all_linear_names, compute_metrics
import torch
from torch import nn
from torch.nn import functional as F
CFG = {
    'seed': 42,
    'train_csv': '/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv',
    'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'max_len': 3096,
    'batch_size': 1,
    'num_classes': 3,
    'model_dir': '/home/mithil/PycharmProjects/lmsys-scoring/models/Meta-Llama-3-8B-Instruct-2560-2-epoch-all-logits',
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
        outputs = model(tensors=inputs)
        logits = outputs.get('logits')
        loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def tokenize_function(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="longest", max_length=max_length)
    return tokenized_inputs


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )




class LLamaClassifier(LlamaPreTrainedModel):
    def __init__(self, model, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3)

    def forward(self, tensors,**kwargs):
        outputs = self.model(**tensors, return_dict=True)
        hidden_states = outputs['logits']
        hidden_states = mean_pooling(hidden_states, tensors['attention_mask']).type(torch.bfloat16)

        return {"logits": self.linear_head(hidden_states)}


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

    train_dataset = Dataset.from_dict({"text": train_df['text'], "targets": train_df['label']})
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

    )

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], trust_remote_code=True,
                                                 attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.float16, quantization_config=quant_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = LLamaClassifier(model,torch_dtype=torch.bfloat16)
    print("Linear layers: ", find_all_linear_names(model))
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=find_all_linear_names(model),
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["linear_head"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    for name, param in model.linear_head.named_parameters():
        param.requires_grad = True


    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, cfg['max_len']))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, )
    train_dataset = train_dataset.remove_columns(["text"])
    print(train_dataset)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

    )

    trainer.train()
    print(model)
    trainer.save_model(cfg['model_dir'])

if __name__ == '__main__':
    main(CFG)
