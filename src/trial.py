from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel
import torch
from peft import PeftModel, LoraConfig, TaskType
from torch import nn

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                             torch_dtype=torch.float16, device_map="auto",
                                             attn_implementation="flash_attention_2")


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class LLamaClassifier(LlamaPreTrainedModel):
    def __init__(self, model):
        super().__init__(config=model.config)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear = nn.Linear(model.config.hidden_size, 3).cuda()

    def forward(self, tensors):
        outputs = self.model(**tensors, return_dict=True)
        hidden_states = outputs['logits']
        hidden_states = mean_pooling(hidden_states, tensors['attention_mask'])

        return {"logits": self.linear(hidden_states)}


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


LLamaClassifierModel = LLamaClassifier(model)
model = PeftModel(LLamaClassifierModel, peft_config=LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules=find_all_linear_names(LLamaClassifierModel),
    task_type=TaskType.SEQ_CLS,
))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
p = tokenizer("Hello, my dog is cute", return_tensors="pt")
for k, v in p.items():
    p[k] = v.to("cuda:0")
output = model(p)
print(output)
model = model.merge_and_unload()
