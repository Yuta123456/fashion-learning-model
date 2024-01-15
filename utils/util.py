import copy
import json
import random
import pandas as pd
import glob
import os
import sys
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from itertools import product

from utils.versatility import calc_increase_v, calc_versatility

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def is_target_category(garment):
    garment_kind = garment["category x color"].split(" × ")[0]
    if garment_kind in [
        "ジャケット",
        "トップス",
        "コート",
        "ニット",
        "タンクトップ",
        "ブラウス",
        "Tシャツ",
        "カーディガン",
        "ダウンジャケット",
        "パーカー",
    ]:
        return True
        # , "ショートパンツ"入れ忘れた
    if garment_kind in ["スカート", "ロングスカート", "ロングパンツ", "ショートパンツ"]:
        return True

    if garment_kind in ["ブーツ", "パンプス", "スニーカー", "靴", "サンダル"]:
        return True

    return False


def calculate_centroid_and_average_distance(tensor_list):
    # 3つのテンソルの重心ベクトルを計算
    centroid = sum(tensor_list) / len(tensor_list)

    # 重心ベクトルに対する各ベクトルのユークリッド距離を計算し合計
    total_distance = 0.0
    for tensor in tensor_list:
        distance = torch.norm(tensor - centroid)  # ユークリッド距離の計算
        total_distance += distance

    # 平均距離を計算
    average_distance = total_distance / len(tensor_list)

    return average_distance


def calculate_euclid_sum(tensor_list):
    distance12 = torch.norm(tensor_list[0] - tensor_list[1])
    distance13 = torch.norm(tensor_list[0] - tensor_list[2])
    distance23 = torch.norm(tensor_list[1] - tensor_list[2])

    # ユークリッド距離の合計を計算
    total_distance = distance12 + distance13 + distance23
    # print(distance12, distance23, distance13)
    return total_distance


def calculate_euclid_max(tensor_list):
    distance12 = torch.norm(tensor_list[0] - tensor_list[1])
    distance13 = torch.norm(tensor_list[0] - tensor_list[2])
    distance23 = torch.norm(tensor_list[1] - tensor_list[2])

    # ユークリッド距離の合計を計算
    max_distance = max(distance12, distance13, distance23)

    return max_distance


def calc_roc_auc(label, score, name):
    fpr, tpr, _ = roc_curve(label, score)

    auc = roc_auc_score(label, score)
    plt.clf()
    plt.plot(fpr, tpr, marker="o")
    plt.xlabel("FPR: False positive rate")
    plt.ylabel("TPR: True positive rate")
    plt.grid()

    plt.text(
        0.5,
        0.9,
        f"AUC = {auc:.10f}",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.savefig(
        f"C:/Users/yuuta/Documents/fashion/model_learning/compatibility/result/{name}_roc.png"
    )
    return auc


def is_include_basic_items(items):
    flag = 0
    for garment in items:
        garment_kind = garment["category x color"].split(" × ")[0]
        if garment_kind in [
            "ジャケット",
            "トップス",
            "コート",
            "ニット",
            "タンクトップ",
            "ブラウス",
            "Tシャツ",
            "カーディガン",
            "ダウンジャケット",
            "パーカー",
        ]:
            flag |= 1
            # , "ショートパンツ"入れ忘れた
        if garment_kind in ["スカート", "ロングスカート", "ロングパンツ", "ショートパンツ"]:
            flag |= 1 << 1

        if garment_kind in ["ブーツ", "パンプス", "スニーカー", "靴", "サンダル"]:
            flag |= 1 << 2
    return flag == 7


def filter_basic_items(items):
    flag = 0
    new_items = []
    for garment in items:
        garment_kind = garment["category x color"].split(" × ")[0]
        # print(garment_kind)
        if (
            garment_kind
            in [
                "ジャケット",
                "トップス",
                "コート",
                "ニット",
                "タンクトップ",
                "ブラウス",
                "Tシャツ",
                "カーディガン",
                "ダウンジャケット",
                "パーカー",
            ]
            and (flag & 1) == 0
        ):
            flag |= 1
            new_items.append(garment)
            # , "ショートパンツ"入れ忘れた
        elif garment_kind in ["スカート", "ロングスカート", "ロングパンツ"] and (flag & (1 << 1)) == 0:
            flag |= 1 << 1
            new_items.append(garment)
        elif garment_kind in ["ブーツ", "パンプス", "スニーカー", "靴", "サンダル"] and (
            flag & (1 << 2) == 0
        ):
            flag |= 1 << 2
            new_items.append(garment)
        # print(flag, garment, new_items)
    return new_items


def get_progress_percent(count, length, div=20):
    progress = count // (length // div)
    res = "\r" + f'【{"*" * progress}{" " * (div - progress)}】'
    return res


def open_json(json_path):
    try:
        with open(json_path, encoding="shift-jis") as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return None


def can_open_images(image_paths):
    try:
        for ip in image_paths:
            Image.open(f"C:/Users/yuuta/Documents/fashion/data/images/{ip}.jpg")
    except Exception as e:
        return False
    return True


def get_category(garment):
    garment_kind = garment["category x color"].split(" × ")[0]
    if garment_kind in [
        "ジャケット",
        "トップス",
        "コート",
        "ニット",
        "タンクトップ",
        "ブラウス",
        "Tシャツ",
        "カーディガン",
        "ダウンジャケット",
        "パーカー",
    ]:
        return "tops"
        # , "ショートパンツ"入れ忘れた
    if garment_kind in ["スカート", "ロングスカート", "ロングパンツ"]:
        return "bottoms"

    if garment_kind in ["ブーツ", "パンプス", "スニーカー", "靴", "サンダル"]:
        return "shoes"


def show_item(itemId):
    # 画像ファイルのパスを指定
    image_path = f"C:/Users/yuuta/Documents/fashion/data/images/{itemId}.jpg"

    # 画像を開いて表示
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.axis("off")  # 軸を非表示にする（オプション）
    plt.show()


# cwのスコアを計算。cwはitemIdで構成され、ver, comはそれぞれidからvectorへの変換器
def calc_cw_score(
    cw: dict[str, list[str]], ver: dict[str, torch.Tensor], com: dict[str, torch.Tensor]
):
    categories = list(cw.keys())
    ver_score = 0
    com_score = 0

    # versatilityの計算
    for cat in categories:
        # print(cw[cat])
        # categoryのアイテム集合
        vectors = [torch.tensor(ver[itemId]) for itemId in cw[cat]]
        ver_score += len(calc_versatility(vectors))

    # compatibilityの計算
    coordinates = list(product(*[cw[c] for c in categories]))
    # print(len(coordinates))
    # 組み合わせを表示
    for coordinate in coordinates:
        vectors = [torch.tensor(com[itemId]) for itemId in coordinate]
        com_score += calculate_euclid_max(vectors)

    # return 1 / com_score.item(), ver_score
    return scale_compatibility(1 / com_score.item()), scale_versability(ver_score)


def scale_compatibility(c):
    # min-max scaling
    min_value = 7.839434299071509e-05
    max_value = 0.0003497860890687236
    standardized_data = (c - min_value) / (max_value - min_value)
    scaled_data = max(0, standardized_data)

    return scaled_data


def scale_versability(v):
    # min-max scaling
    min_value = 642.0
    max_value = 1030.0
    standardized_data = (v - min_value) / (max_value - min_value)
    scaled_data = max(0, standardized_data)

    return scaled_data


# compatibilityの追加スコア
def calc_increase_c(
    cw: dict[str, list[str]], com: dict[str, torch.Tensor], item: str, category: str
):
    categories = list(cw.keys())
    com_score = 0
    new_cw = copy.deepcopy(cw)
    new_cw[category] = [item]
    # compatibilityの計算
    coordinates = list(product(*[new_cw[c] for c in categories]))
    # print(len(coordinates))

    # 組み合わせを表示
    for coordinate in coordinates:
        vectors = [torch.tensor(com[itemId]) for itemId in coordinate]
        com_score = calculate_euclid_max(vectors)

    return 1 / com_score


def calc_compatibility_score(cw: dict[str, list[str]], com: dict[str, torch.Tensor]):
    categories = list(cw.keys())
    com_score = 0
    # compatibilityの計算
    coordinates = list(product(*[cw[c] for c in categories]))

    for coordinate in coordinates:
        vectors = [torch.tensor(com[itemId]) for itemId in coordinate]
        com_score += calculate_euclid_max(vectors)

    return scale_compatibility(1 / com_score.item())


def optimize_cw(
    cw: dict[str, list[str]],
    dataset: dict[str, list[str]],
    com: dict[str, torch.Tensor],
    ver: dict[str, torch.Tensor],
):
    categories = list(cw.keys())
    # 枚数
    T = 3
    eps = 10e-3
    increase_score = eps + 1
    pre_cw_score = 0
    while increase_score > eps:
        cw_score_confirm = 0
        for cat in categories:
            # そのレイヤを空にする
            cw[cat] = []
            # それぞれのスコアを初期化
            ver_set = set()
            com_score = 0
            for _ in range(T):
                # 追加するアイテムと、そのアイテムが追加されたことによるCWスコアの上昇値を初期化
                add_item = None
                max_increase_score = 10e-5
                for item in dataset[cat]:
                    if item in cw[cat]:
                        continue
                    # 増加分を計算
                    increase_c = calc_increase_c(cw, com, item, cat)
                    cover_clusters = calc_increase_v(torch.tensor(ver[item]))
                    increase_v = len(cover_clusters.union(ver_set)) - len(ver_set)

                    increase = increase_c + increase_v
                    if max_increase_score < increase:
                        max_increase_score = increase
                        add_item = item

                com_score += calc_increase_c(cw, com, add_item, cat)
                ver_set = ver_set.union(calc_increase_v(torch.tensor(ver[add_item])))
                # print(com_score)
                cw[cat].append(add_item)
            # cw_score_confirm += com_score + len(ver_set)
            cw_score_confirm += com_score
        score = sum(calc_cw_score(cw, com, ver))
        increase_score = score - pre_cw_score
        pre_cw_score = score
