from transformers import AutoModelForSequenceClassification, AutoConfig
from torch import nn


class WinnerModel(nn.Module):
    def __init__(self, model_name, num_labels, tokenizer):
        super(WinnerModel, self).__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, inputs):
        return self.model(**inputs).logits