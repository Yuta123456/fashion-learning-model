import torch.nn as nn
from transformers import AutoModel

class CaptionEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-v2")
  def forward(self, x):
    x = self.bert(x)
    x = x.last_hidden_state
    x = x[:,0,:] 
    return x