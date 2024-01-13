import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import tomotopy as tp

sys.path.append("C:/Users/yuuta/Documents/fashion")

from model_learning.model_structure.image_encoder import (
    ImageEncoder,
    ImageEncoderV2,
    ImageEncoderV3,
)

if torch.cuda.is_available():
    device = torch.device("cuda")  # GPUデバイスを取得
else:
    device = torch.device("cpu")  # CPUデバイスを取得


model = ImageEncoderV2().to(device)

model.load_state_dict(
    torch.load(
        "C:/Users/yuuta/Documents/fashion/model_learning/compatibility/models/triplet-image-2024-01-09.pth"
    )
)

v_model = ImageEncoderV2().to(device)

v_model.load_state_dict(
    torch.load(
        "C:/Users/yuuta/Documents/fashion/model_learning/versatility/models/triplet-image-2024-01-11.pth"
    )
)

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def id_to_vector(itemId):
    image_path = f"C:/Users/yuuta/Documents/fashion/data/images/{itemId}.jpg"
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("L")
        image = Image.merge("RGB", [image] * 3)
    input_img = transform(image).to(device)
    input_img = torch.unsqueeze(input_img, 0)
    with torch.no_grad():
        pred = model(input_img)

    return pred


def topic_model_infer(mdl, attributes):
    inf_doc = mdl.make_doc(attributes)

    log_prob = mdl.infer(inf_doc, iter=500)
    # print(log_prob)
    return log_prob[1]


def id_to_vector_in_versatility(itemId):
    image_path = f"C:/Users/yuuta/Documents/fashion/data/images/{itemId}.jpg"
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("L")
        image = Image.merge("RGB", [image] * 3)
    input_img = transform(image).to(device)
    input_img = torch.unsqueeze(input_img, 0)
    with torch.no_grad():
        pred = v_model(input_img)

    return pred
