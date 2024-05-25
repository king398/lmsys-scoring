from transformers import AutoTokenizer, MistralForSequenceClassification
import torch
from peft import PeftModel, LoraConfig
from utils import find_all_linear_names

base_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/Promethus-eval-2560-2-epoch-classification"

model = MistralForSequenceClassification.from_pretrained("prometheus-eval/prometheus-7b-v2.0",
                                                         torch_dtype=torch.float16, device_map="auto", num_labels=3)
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules=find_all_linear_names(model),
    task_type="SEQ_CLS",
)

model = PeftModel.from_pretrained(model, base_path, config=peft_config)
model = model.merge_and_unload()
model.save_pretrained(f"{base_path}/merged")
tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
tokenizer.save_pretrained(f"{base_path}/merged")
