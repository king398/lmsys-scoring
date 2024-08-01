from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
model = AutoModelForSequenceClassification.from_pretrained("unsloth/gemma-2-27b-it-bnb-4bit", device_map="auto",
                                                           torch_dtype=torch.bfloat16,num_labels=3,attn_implementation="flash_attention_2")


tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-27b-it-bnb-4bit")


trial = "This is a trial prompt"
inputs = tokenizer(trial, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
print(model)