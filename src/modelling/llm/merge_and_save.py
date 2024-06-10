from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel, LoraConfig
from utils import find_all_linear_names

base_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/Meta-Llama-3-8B-Instruct-2560-2-epoch-extra-data-lmsys"

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                         torch_dtype=torch.float16, device_map="auto", )

model = PeftModel.from_pretrained(model, base_path)
model = model.merge_and_unload()
model.save_pretrained(f"{base_path}/merged")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.save_pretrained(f"{base_path}/merged")
