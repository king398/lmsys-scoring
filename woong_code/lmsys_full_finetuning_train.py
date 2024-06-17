
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
import matplotlib.pyplot as plt
from PIL import Image
import requests
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator

from datasets import load_dataset 
import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer, DataCollatorWithPadding
from transformers import AutoTokenizer, LlamaModel, LlamaForSequenceClassification
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss
import torch.nn.functional as F
from accelerate import Accelerator

def softmax(logits):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return  ' '.join(sentences)
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
        # Ensure label_weights is a tensor
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.bfloat16).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels and convert them to long type for cross_entropy
        labels = inputs.pop("labels").long()

        # Forward pass
        outputs = model(**inputs)

        # Extract logits assuming they are directly outputted by the model
        logits = outputs.get('logits')

        # Compute custom loss with class weights for imbalanced data handling
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

MAX_LEN=1024

os.environ["TRANSFORMERS_CACHE"] = "/mnt/alignment-handbook/weights"
cache_dir = "/mnt/alignment-handbook/weights"
os.makedirs("/mnt/alignment-handbook/weights", exist_ok=True)
train = pd.read_csv("/mnt/alignment-handbook/LMSYS/train.csv")


train.loc[:, 'prompt'] = train['prompt'].apply(process)
train.loc[:, 'response_a'] = train['response_a'].apply(process)
train.loc[:, 'response_b'] = train['response_b'].apply(process)

train['text'] = 'User prompt: ' + train['prompt'] +  '\n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n'  + train['response_b']
## i want to length of average text length, max length, min length
train['text_len'] = train['text'].apply(lambda x: len(x.split()))
#print(train['text_len'].describe(), train['text_len'].quantile(0.99), train['text_len'].quantile(0.95), train['text_len'].quantile(0.90))
#train = train[train['text_len'] < 1024]
#print(len(train))
#import sys
#sys.exit()
train['target'] = np.argmax(train[['winner_model_a','winner_model_b','winner_tie']].values, axis=1)

model_string = train['model_a'].unique().tolist()
train['model_a'] = train['model_a'].apply(lambda x: model_string.index(x))
train['model_b'] = train['model_b'].apply(lambda x: model_string.index(x))
# columns : [id,model_a,model_b,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie]
# model_a랑 model_b로 multilabel로 나눈다
model_labels = train[['model_a','model_b']].values
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
train['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(mskf.split(train, model_labels)):
    train['fold'].iloc[valid_idx] = i
fold = 0
train_df = train[train['fold'] != fold].reset_index(drop=True)
valid_df = train[train['fold'] == fold].reset_index(drop=True)
train_df.drop(['fold','model_a','model_b','id','response_a','response_b','prompt','winner_model_a','winner_model_b','winner_tie'], axis=1, inplace=True)
valid_df.drop(['fold','model_a','model_b','id','response_a','response_b','prompt','winner_model_a','winner_model_b','winner_tie'], axis=1, inplace=True)
dataset_train = datasets.Dataset.from_pandas(train_df)
dataset_valid = datasets.Dataset.from_pandas(valid_df)
dataset = datasets.DatasetDict({'train': dataset_train, 'val': dataset_valid})

class_weights=(1/train_df.target.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()
#class_weights=None


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


# model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#         use_flash_attention_2=model_args.use_flash_attention_2,
#         torch_dtype=torch_dtype,
#         use_cache=False if training_args.gradient_checkpointing else True,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#         cache_dir=cache_dir,
#     )

model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
)

model.config.pad_token_id = tokenizer.pad_token_id

def llama_preprocessing_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)
    
tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("target", "labels")

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

# args = TrainingArguments(
#     num_train_epochs=1,
#     remove_unused_columns=False,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     gradient_accumulation_steps=4,
#     warmup_steps=1200,
#     learning_rate=2e-2,
#     weight_decay=1e-6,
#     adam_beta2=0.999,
#     logging_steps=100,
#     optim="adamw_hf",
#     save_strategy="steps",
#     save_steps=1000,
#     evaluation_strategy="steps",
#     eval_steps=500,
#     push_to_hub=True,
#     save_total_limit=1,
#     bf16=True,
#     report_to=["tensorboard", "wandb"],
#     dataloader_pin_memory=False,
#     output_dir="paligemma-vqa"
# )
training_args = TrainingArguments(
    learning_rate = 4e-6,
    remove_unused_columns = True,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 4,
    warmup_steps=160,
    num_train_epochs = 2,
    weight_decay = 1e-6,
    evaluation_strategy = 'steps',
    save_strategy = 'steps',
    save_steps=400,
    eval_steps=200,
    save_total_limit=1,
    report_to=["wandb"],
    dataloader_pin_memory=True,
    load_best_model_at_end = True,
    bf16=True,
    output_dir="paligemma_rec_fullfinetuning",
    push_to_hub=True,
    #dataloader_num_workers=16,
    #packing option
)

trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['val'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    class_weights=class_weights,
)
train_result = trainer.train()
