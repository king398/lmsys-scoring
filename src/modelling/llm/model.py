import torch
from torch import nn
from transformers import LlamaPreTrainedModel, MistralPreTrainedModel, Gemma2PreTrainedModel
import torch.nn.functional as F


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
        self.linear_head = nn.Linear(model.config.hidden_size * 2, 3)

        self.dtype_linear = torch.bfloat16

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, **kwargs):
        outputs_1 = self.model(input_ids=input_ids_1, attention_mask=input_ids_1, return_dict=True)
        hidden_states_1 = outputs_1['logits']
        hidden_states_1 = mean_pooling(hidden_states_1, attention_mask_1).type(torch.bfloat16)
        outputs_2 = self.model(input_ids=input_ids_2, attention_mask=input_ids_2, return_dict=True)
        hidden_states_2 = outputs_2['logits']
        hidden_states_2 = mean_pooling(hidden_states_2, attention_mask_2).type(torch.bfloat16)
        hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=-1)

        return {"logits": self.linear_head(hidden_states)}


class MistralClassifier(MistralPreTrainedModel):
    def __init__(self, model, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size * 2, 3)

        self.dtype_linear = torch.bfloat16

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, **kwargs):
        outputs_1 = self.model(input_ids=input_ids_1, attention_mask=input_ids_1, return_dict=True)
        hidden_states_1 = outputs_1['logits']
        hidden_states_1 = mean_pooling(hidden_states_1, attention_mask_1).type(torch.bfloat16)
        outputs_2 = self.model(input_ids=input_ids_2, attention_mask=input_ids_2, return_dict=True)
        hidden_states_2 = outputs_2['logits']
        hidden_states_2 = mean_pooling(hidden_states_2, attention_mask_2).type(torch.bfloat16)
        hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=-1)

        return {"logits": self.linear_head(hidden_states)}


class GemmaClassifier(Gemma2PreTrainedModel):
    def __init__(self, model, **kwargs):
        super().__init__(config=model.config, **kwargs)
        self.model = model
        self.model.lm_head = nn.Identity()
        self.linear_head = nn.Linear(model.config.hidden_size * 2, 3)

        self.dtype_linear = torch.bfloat16

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, **kwargs):
        outputs_1 = self.model(input_ids=input_ids_1, attention_mask=input_ids_1, return_dict=True)
        hidden_states_1 = outputs_1['logits']
        hidden_states_1 = mean_pooling(hidden_states_1, attention_mask_1).type(torch.bfloat16)
        outputs_2 = self.model(input_ids=input_ids_2, attention_mask=input_ids_2, return_dict=True)
        hidden_states_2 = outputs_2['logits']
        hidden_states_2 = mean_pooling(hidden_states_2, attention_mask_2).type(torch.bfloat16)
        hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=-1)

        return {"logits": self.linear_head(hidden_states)}
