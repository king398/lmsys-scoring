import torch
from torch import nn
from transformers import LlamaPreTrainedModel, MistralPreTrainedModel, Gemma2PreTrainedModel, Phi3PreTrainedModel
import torch.nn.functional as F
from modeling_internlm2 import InternLM2PreTrainedModel


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, attention_mask):
        scores = self.attention_weights(x)
        scores = scores.squeeze(-1)

        scores = scores.masked_fill(attention_mask == 0, -float('inf'))

        attn_weights = F.softmax(scores, dim=-1)

        attn_weights = attn_weights.unsqueeze(-1)
        pooled_output = torch.sum(x * attn_weights, dim=1)

        return pooled_output


class LLamaClassifier(LlamaPreTrainedModel):
    def __init__(self, model, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3)

        self.dtype_linear = torch.bfloat16

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs['logits']
        hidden_states = mean_pooling(hidden_states, attention_mask).type(torch.bfloat16)

        return {"logits": self.linear_head(hidden_states)}


class PhiClassifier(Phi3PreTrainedModel):
    def __init__(self, model, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        # self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3)

        self.dtype_linear = torch.bfloat16

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                             output_hidden_states=True)
        hidden_states = outputs['hidden_states'][-1]
        hidden_states = mean_pooling(hidden_states, attention_mask).type(torch.bfloat16)

        return {"logits": self.linear_head(hidden_states)}


class MistralClassifier(MistralPreTrainedModel):
    def __init__(self, model, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3)

        self.dtype_linear = torch.bfloat16

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs['logits']
        hidden_states = mean_pooling(hidden_states, attention_mask).type(torch.bfloat16)

        return {"logits": self.linear_head(hidden_states)}


class GemmaClassifier(Gemma2PreTrainedModel):
    def __init__(self, model, dtype_linear=torch.bfloat16, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size, 3)

        self.dtype_linear = dtype_linear

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs['logits']
        hidden_states = mean_pooling(hidden_states, attention_mask).type(self.dtype_linear)

        return {"logits": self.linear_head(hidden_states), }
