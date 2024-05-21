from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
base_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/Prometheus-eval-2-epoch-2560-len"

model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-7b-v2.0", torch_dtype=torch.float16,device_map="auto")
model = PeftModel.from_pretrained(model, base_path)
model = model.merge_and_unload()
model.save_pretrained(f"{base_path}/merged")
tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
tokenizer.save_pretrained(f"{base_path}/merged")