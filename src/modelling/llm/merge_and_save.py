from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel
import torch
from peft import PeftModel
from torch import nn
from model import GemmaClassifier
base_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/gemma-2-9b-it-retrained"

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it",
                                             torch_dtype=torch.float16, device_map="cuda:1", )
#model = GemmaClassifier(model).to("cuda:1")
#model = PeftModel.from_pretrained(model, base_path)
#model = model.merge_and_unload()
#model.save_pretrained(f"{base_path}/merged")
#tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
#tokenizer.save_pretrained(f"{base_path}/merged")

# trial prompt
model = GemmaClassifier.from_pretrained(model).to("cuda:1")
model.from_pretrained()