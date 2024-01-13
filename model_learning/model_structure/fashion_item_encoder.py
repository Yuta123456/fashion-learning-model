import torch.nn as nn
import torch
from transformers import AutoTokenizer
from model_learning.model_structure.caption_encode import CaptionEncoder

from model_learning.model_structure.image_encoder import ImageEncoder


class FashionItemEncoder(nn.Module):
    def __init__(self, device):
        super(FashionItemEncoder, self).__init__()
        self.device = device
        self.image_model = ImageEncoder(768)
        self.caption_model = CaptionEncoder()
        self.fc1 = nn.Linear(768 * 2, 768)
        self.fc2 = nn.Linear(768, 324)
        self.fc3 = nn.Linear(324, 64)
        self.relu = nn.ReLU()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-v2"
        )

    def load_image_dict(self, image_model_path: str):
        self.image_model.load_state_dict(torch.load(image_model_path))

    def load_caption_dict(self, caption_model_path: str):
        self.caption_model.load_state_dict(torch.load(caption_model_path))

    def forward(self, image, caption):
        image_vector = self.image_model(image)
        ids = self.tokenizer.batch_encode_plus(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        ).input_ids
        ids = ids.to(self.device)
        caption_vector = self.caption_model(ids)
        concat_vector = torch.cat((image_vector, caption_vector), dim=1)
        concat_vector = self.relu(concat_vector)
        y = self.fc1(concat_vector)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        return y
