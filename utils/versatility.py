import torch

centroids = torch.load(
    f"C:/Users/yuuta/Documents/fashion/model_learning/versatility/cluster/center_tensors_400.pt"
).numpy()
NEAREST_ITEM_LENGTH = 210


def calc_versatility(vectors: list[torch.Tensor]):
    covered_centroid_ids = set()
    for v in vectors:
        distances = torch.norm(v - centroids, p=2, dim=1)
        sorted_indices = torch.argsort(distances)
        k_min_indices = sorted_indices[:NEAREST_ITEM_LENGTH]
        labels: set[int] = set(k_min_indices.tolist())
        covered_centroid_ids = covered_centroid_ids.union(labels)

    return covered_centroid_ids


def calc_increase_v(vector: torch.Tensor):
    distances = torch.norm(vector - centroids, p=2, dim=1)
    sorted_indices = torch.argsort(distances)
    k_min_indices = sorted_indices[:NEAREST_ITEM_LENGTH]
    labels: set[int] = set(k_min_indices.tolist())

    return set(labels)
