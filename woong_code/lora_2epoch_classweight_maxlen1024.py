import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from accelerate import Accelerator
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Training script with dynamic fold setting")
    parser.add_argument("--fold", type=int, default=0, help="Specify the fold to use for training and validation")
    return parser.parse_args()

# Custom functions
def softmax(logits):
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return ' '.join(sentences)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pred_classes = np.argmax(predictions, axis=1)
    return {
        'balanced_accuracy': balanced_accuracy_score(labels, pred_classes),
        'accuracy': accuracy_score(labels, pred_classes),
        'log_loss': log_loss(labels, softmax(predictions))
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

MAX_LEN = 1024

# Main function
def main():
    args = parse_args()

    os.environ["TRANSFORMERS_CACHE"] = "/mnt/alignment-handbook/weights"
    cache_dir = "/mnt/alignment-handbook/weights"
    os.makedirs("/mnt/alignment-handbook/weights", exist_ok=True)
    train = pd.read_csv("/mnt/alignment-handbook/LMSYS/train.csv")

    train.loc[:, 'prompt'] = train['prompt'].apply(process)
    train.loc[:, 'response_a'] = train['response_a'].apply(process)
    train.loc[:, 'response_b'] = train['response_b'].apply(process)

    train['text'] = 'User prompt: ' + train['prompt'] +  '\n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n'  + train['response_b']
    train['text_len'] = train['text'].apply(lambda x: len(x.split()))
    train['target'] = np.argmax(train[['winner_model_a','winner_model_b','winner_tie']].values, axis=1)

    model_string = train['model_a'].unique().tolist()
    train['model_a'] = train['model_a'].apply(lambda x: model_string.index(x))
    train['model_b'] = train['model_b'].apply(lambda x: model_string.index(x))

    model_labels = train[['model_a','model_b']].values
    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train['fold'] = -1
    for i, (train_idx, valid_idx) in enumerate(mskf.split(train, model_labels)):
        train['fold'].iloc[valid_idx] = i

    fold = args.fold
    train_df = train[train['fold'] != fold].reset_index(drop=True)
    valid_df = train[train['fold'] == fold].reset_index(drop=True)
    train_df.drop(['fold','model_a','model_b','id','response_a','response_b','prompt','winner_model_a','winner_model_b','winner_tie'], axis=1, inplace=True)
    valid_df.drop(['fold','model_a','model_b','id','response_a','response_b','prompt','winner_model_a','winner_model_b','winner_tie'], axis=1, inplace=True)
    dataset_train = Dataset.from_pandas(train_df)
    dataset_valid = Dataset.from_pandas(valid_df)
    dataset = DatasetDict({'train': dataset_train, 'val': dataset_valid})

    class_weights = (1/train_df.target.value_counts(normalize=True).sort_index()).tolist()
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights / class_weights.sum()
    #class_weights = None

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    from transformers import BitsAndBytesConfig
    from accelerate import Accelerator

    current_device = Accelerator().process_index
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.10,
        bias='none',
        task_type=TaskType.SEQ_CLS,
        target_modules=['o_proj', 'v_proj']
    )

    model = LlamaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map={"": current_device}
    )
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id

    def llama_preprocessing_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)
    
    tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("target", "labels")
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        learning_rate=1e-4,
        remove_unused_columns=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=240,
        num_train_epochs=2,
        weight_decay=1e-6,
        evaluation_strategy='steps',
        save_strategy='steps',
        save_steps=400,
        eval_steps=200,
        save_total_limit=1,
        report_to=["wandb"],
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        bf16=True,
        output_dir=f"paligemma_rec_lora_kalora_test_fold{args.fold}_classweight",
        push_to_hub=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    train_result = trainer.train()

if __name__ == "__main__":
    main()