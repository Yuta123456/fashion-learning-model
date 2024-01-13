import torch

"""
セントロイドとアイテムの埋め込みを受け取って、一番近いラベルのsetを返す関数
"""


def calc_versatility(
    centroids: list[torch.Tensor], vectors: list[torch.Tensor], NEAREST_ITEM_LENGTH=30
):
    covered_centroid_ids = set()
    for v in vectors:
        # v = v.expand(centroids.size)
        # print(v.shape, centroids.shape)
        distances = torch.norm(v - centroids, p=2, dim=1)
        # print(distances.shape)
        sorted_indices = torch.argsort(distances)
        k_min_indices = sorted_indices[:NEAREST_ITEM_LENGTH]
        labels: set[int] = set(k_min_indices.tolist())
        covered_centroid_ids = covered_centroid_ids.union(labels)
        # print(covered_centroid_ids)

    return covered_centroid_ids
