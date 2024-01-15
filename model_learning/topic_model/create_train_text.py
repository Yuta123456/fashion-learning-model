import glob
import json

from sklearn.metrics import *
import pandas as pd
import sys

sys.path.append("C:/Users/yuuta/Documents/fashion")
from utils.util import filter_basic_items, get_progress_percent, is_target_category

coordinates_file = glob.glob(
    f"C:/Users/yuuta/Documents/fashion/data/train/**/*_new.json"
)

with open(
    "C:/Users/yuuta/Documents/fashion/data/attributes_train.json",
    "r",
    encoding="utf-8",
) as f:
    attributes = json.load(f)

documents = ""
SIZE = len(coordinates_file)
for i, fp in enumerate(coordinates_file):
    print(get_progress_percent(i, SIZE), end="")
    json_dict = pd.read_json(fp, encoding="shift-jis")
    coordinate_attributes = ""
    items = filter_basic_items(json_dict["items"])
    if len(items) != 3:
        continue
    for item in items:
        item_id = str(item["itemId"])
        try:
            attribute = attributes[item_id]
        except KeyError as e:
            print(e)
        coordinate_attributes += ", ".join(attribute)
    documents += coordinate_attributes + "\n"

with open(
    "C:/Users/yuuta/Documents/fashion/model_learning/topic_model/train_new.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(documents)
