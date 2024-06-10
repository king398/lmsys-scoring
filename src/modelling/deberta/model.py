from transformers import AutoModel, AutoConfig
from torch import nn
import torch


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class WinnerModel(nn.Module):
    def __init__(self, model_name, num_labels, tokenizer):
        super(WinnerModel, self).__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(model_name, config=config)

        self.model.resize_token_embeddings(len(tokenizer))
        self.pooling = MeanPooling()
        self.fc = nn.Linear(config.hidden_size, num_labels)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mean_pooling = self.pooling(last_hidden_state, attention_mask)
        # cast as float16
        mean_pooling = mean_pooling.bfloat16()
        logits = self.fc(mean_pooling)
        return logits
