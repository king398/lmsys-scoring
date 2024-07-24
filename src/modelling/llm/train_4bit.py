import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from transformers import (
    AutoTokenizer, EvalPrediction, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, Gemma2PreTrainedModel, Gemma2Model
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch.nn as nn
# Configuration
cfg = {
    'wandb_project': 'lmsys-winner',
    'train_csv': "data/train_folds_llama.csv",
    'model_path': "unsloth/gemma-2-9b-it-bnb-4bit",
    'max_length': 1536,
    'target_columns': ['winner_model_a', 'winner_model_b', 'winner_tie'],
    'columns_to_vectorize': ["prompt", "response_a", "response_b"],
    'output_dir': '/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-next-token',
    'model_dir': '/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-next-token'
}

os.environ['WANDB_PROJECT'] = cfg['wandb_project']

def load_and_preprocess_data(csv_path, columns_to_vectorize, target_columns):
    df = pd.read_csv(csv_path)
    df['label'] = df[target_columns].idxmax(axis=1)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    return df[columns_to_vectorize + ['label', 'fold']]

def setup_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_eos_token = True
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_function(example, tokenizer, max_length, label_ids):
    prompt = eval(example['prompt'], {"null": ['']})
    response_a = eval(example['response_a'], {"null": ['']})
    response_b = eval(example['response_b'], {"null": ['']})

    text = "Which one of the chatbots below did a better job responding to the user request? Or were they tied? \n"
    text += "~~~~~~~~~~ CONVERSATION WITH BOT A ~~~~~~~~~~\n"
    text += "".join([f"""### User: "{p}" ### Bot A Response: "{r}""" for p, r in zip(prompt, response_a)])
    text += "\n~~~~~~~~~~ CONVERSATION WITH BOT B ~~~~~~~~~~\n"
    text += "".join([f"""### User: "{p}" ### Bot B Response: "{r}""" for p, r in zip(prompt, response_b)])
    text += "\n ### BEST RESPONSE: "

    label_token_id = label_ids[int(example['label'])]
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    if len(input_ids) >= max_length - 2 - len([label_token_id]):
        input_ids = input_ids[:max_length - 2 - len([label_token_id])]

    input_ids = [tokenizer.bos_token_id] + input_ids + [label_token_id] + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * (len(input_ids) - 2) + [label_token_id, tokenizer.eos_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def load_dataset(df, tokenizer, max_length, label_ids):
    dataset = Dataset.from_pandas(df)
    return dataset.map(
        lambda example: tokenize_function(example, tokenizer, max_length, label_ids),
        remove_columns=dataset.column_names
    )

def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    label_tokens_ids = np.array(LABEL_IDS)
    index_mapping = {value.item(): idx for idx, value in enumerate(label_tokens_ids)}
    labels = labels[np.isin(labels, label_tokens_ids)]
    labels = np.array([index_mapping[label.item()] for label in labels])
    acc = accuracy_score(labels, preds)
    probs = softmax(logits, axis=-1)
    log_loss_value = log_loss(labels, probs)
    return {'accuracy': acc, 'log_loss': log_loss_value}

class Gemma2ForSFT(Gemma2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Gemma2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, cache_position=None):
        outputs = self.model(
            input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
            use_cache, output_attentions, output_hidden_states, return_dict, cache_position
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()

            label_tokens_ids = torch.tensor(LABEL_IDS).to(self.device)
            index_mapping = {value.item(): idx for idx, value in enumerate(label_tokens_ids)}
            true_labels = shift_labels[torch.isin(shift_labels, label_tokens_ids)]
            true_labels = torch.tensor([index_mapping[label.item()] for label in true_labels]).to(self.device)
            true_logits = shift_logits[torch.isin(shift_labels, label_tokens_ids)][:, label_tokens_ids].to(self.device)
            print(true_logits,true_labels)
            loss = loss_fct(true_logits, true_labels)

        return CausalLMOutputWithPast(loss=loss, logits=true_logits)

def setup_model(model_path):
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
        task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'k_proj', 'v_proj'],
        inference_mode=False
    )
    model = Gemma2ForSFT.from_pretrained(model_path, torch_dtype=torch.bfloat16, load_in_4bit=True,device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    return model

def main():
    df = load_and_preprocess_data(cfg['train_csv'], cfg['columns_to_vectorize'], cfg['target_columns'])
    tokenizer = setup_tokenizer(cfg['model_path'])

    global LABEL_IDS
    LABEL_IDS = [tokenizer(i, add_special_tokens=False)["input_ids"][0] for i in ['a', 'b', 'tie']]

    train_ds = load_dataset(df[df['fold'] != 0], tokenizer, cfg['max_length'], LABEL_IDS)
    eval_ds = load_dataset(df[df['fold'] == 0], tokenizer, cfg['max_length'], LABEL_IDS)

    model = setup_model(cfg['model_path'])
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=cfg['output_dir'],
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=10,
        warmup_steps=20,
        optim="adamw_8bit",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        bf16=True,
        metric_for_best_model="log_loss",
        greater_is_better=False,
        report_to="wandb",
        eval_steps=5000,
        load_best_model_at_end=True,
        run_name=cfg['model_dir'].split("/")[-1],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == '__main__':
    main()