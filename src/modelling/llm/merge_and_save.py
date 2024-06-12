from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel
import torch
from peft import PeftModel
from torch import nn

base_path = "/home/mithil/PycharmProjects/lmsys-scoring/models/Meta-Llama-3-8B-Instruct-2560-2-epoch-all-logits"

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                             torch_dtype=torch.float16, device_map="cuda:0", )


class LLamaClassifier(LlamaPreTrainedModel):
    def __init__(self, model, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3).cuda()

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, tensors, **kwargs):
        outputs = self.model(**tensors, return_dict=True)
        hidden_states = outputs['logits']
        hidden_states = self.mean_pooling(hidden_states, tensors['attention_mask']).type(torch.float16)

        return {"logits": self.linear_head(hidden_states)}


model = LLamaClassifier(model)

model = PeftModel.from_pretrained(model, base_path)
model = model.merge_and_unload()
model.save_pretrained(f"{base_path}/merged")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.save_pretrained(f"{base_path}/merged")
