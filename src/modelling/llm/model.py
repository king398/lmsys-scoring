import torch
from torch import nn
from transformers import LlamaPreTrainedModel
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
        self.linear_head = nn.Linear(model.config.hidden_size, 3)
        self.attention_pooling = AttentionPooling(model.config.hidden_size)
        self.dtype_linear = torch.bfloat16

    def forward(self, tensors, **kwargs):
        outputs = self.model(**tensors, return_dict=True)
        hidden_states = outputs['logits'].type(self.dtype_linear)
        hidden_states = self.attention_pooling(hidden_states, tensors['attention_mask']).type(self.dtype_linear)

        return {"logits": self.linear_head(hidden_states)}
