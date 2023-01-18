from torch.utils.data import default_collate


def collate_double_seg(batch) -> tuple:
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    """
    x = [sample["x"] for sample in batch]
    y = [sample["y"] for sample in batch]
    x_name = [sample["x_name"] for sample in batch]
    return default_collate(x), default_collate(y), x_name