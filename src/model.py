from transformers import AutoModel
import torch
import torch.nn as nn
import omegaconf

class EncoderForClassification(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, model_config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, label, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state 
        pooled = hidden[:, 0, :] 
        logits = self.classifier(pooled)  
        loss = self.loss_fn(logits, label)
        
        return {'logits': logits, 'loss': loss}
