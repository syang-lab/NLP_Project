import torch
from transformers import AutoModel

class DistillBERTClass(torch.nn.Module):
    def __init__(self, checkpoint_model):
        super(DistillBERTClass, self).__init__()
        self.pre_trained_model = AutoModel.from_pretrained(checkpoint_model)
        self.linear = torch.nn.Linear(768, 768)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 12)

    def forward(self, input_ids, attention_mask):
        pre_trained_output = self.pre_trained_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = pre_trained_output.last_hidden_state

        hidden_state = hidden_state[:, 0, :]
        output = self.linear(hidden_state)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.classifier(output)
        return output




