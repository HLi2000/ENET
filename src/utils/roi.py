import json
import pathlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Union
from torch.utils.data import Dataset
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_area, box_convert
from src.metrics.bounding_box import BoundingBox
from src.metrics.enumerators import BBFormat, BBType
import matplotlib.path as pltPath
from torch.utils.data.dataloader import default_collate

def crop_rect(image, box, rect = None):
    """
    Crop squares but added exclusion rules to remove low quality crops, setting the limitation of side length
    (in pixel) to adjust crop qualities

    # Arguments
        image: image read by OpenCV library
        box: box points returned from border following (returned by cv2.boxPoints(rect))
        rect: min area rectangular returned from border following (returned by cv2.minAreaRect())

    # Returns
        return a list contains all the cropped square crops, each crop is stored in a numpy array

    """
    box = np.array(box)
    if len(box.shape) == 1:
        x, y, x2, y2 = box
        warped = image[int(x):int(x2), int(y):int(y2)]
        warped = np.ascontiguousarray(warped, dtype=np.uint8)
    else:
        width = int(rect[1][0])
        height = int(rect[1][1])

        box = np.array(box)
        src_pts = box.astype("float32")

        # coordinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(image, M, (width, height))


    # visualisation for debugging
    # plt.figure(dpi=200)
    # plt.imshow(warped)
    # plt.show()

    return warped

def crop_square(image, box, rect = None):
    """
    Crop squares but added exclusion rules to remove low quality crops, setting the limitation of side length
    (in pixel) to adjust crop qualities

    # Arguments
        image: image read by OpenCV library
        box: box points returned from border following (returned by cv2.boxPoints(rect))
        rect: min area rectangular returned from border following (returned by cv2.minAreaRect())

    # Returns
        return a list contains all the cropped square crops, each crop is stored in a numpy array

    """
    box = np.array(box)
    if len(box.shape) == 1:
        x, y, x2, y2 = box
        warped = image[int(x):int(x2), int(y):int(y2)]
        warped = np.ascontiguousarray(warped, dtype=np.uint8)
        width = warped.shape[1]
        height = warped.shape[0]
    else:
        width = int(rect[1][0])
        height = int(rect[1][1])

        box = np.array(box)
        src_pts = box.astype("float32")

        # coordinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(image, M, (width, height))

    crop_list = []

    x0 = 0
    y0 = 0
    len_threshold = 0.5
    side_threshold = 200

    if width > height:
        L = height
        if L >= side_threshold:
            x1 = x0 + L
            y1 = L
            while x1 <= width - 1:
                crop_list.append(warped[y0:y1, x0:x1])
                # visualisation for debugging
                # cv2.rectangle(warped, (x0, y0), (x1, y1), (0, 255, 0), 20)
                if x1 == width - 1:
                    break
                x1 = x1 + L if x1 + L < width else (width - 1 if width - x1 >= L * len_threshold else width)
                x0 = x1 - L
    else:
        L = width
        if L >= side_threshold:
            x1 = L
            y1 = y0 + L
            while y1 <= height - 1:
                crop_list.append(warped[y0:y1, x0:x1])
                # visualisation for debugging
                # cv2.rectangle(warped, (x0, y0), (x1, y1), (0, 255, 0), 20)
                if y1 == height - 1:
                    break
                y1 = y1 + L if y1 + L < height else (height - 1 if height - y1 >= L * len_threshold else height)
                y0 = y1 - L
    # visualisation for debugging
    # plt.figure(dpi=200)
    # plt.imshow(warped)
    # plt.show()

    return crop_list


def draw_box_countours(mask, rotated=True):
    """
        This function is to draw box contour images from binary narray seg masks.

        return list XYXY or [x,y]*4
    """
    mask = mask.astype(np.uint8) * 255
    mask_img = np.stack([mask, mask, mask], axis=-1)
    boxes = []
    rects = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cntr in contours:
        if rotated:
            rect = cv2.minAreaRect(cntr)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(cntr)
            # Abandon boxes with too small area
            if (area > 1000.0):
                boxes.append(box.tolist())
                rects.append(rect)
                cv2.drawContours(mask_img, [box], 0, (0, 0, 255), 10)
        else:
            x, y, w, h = cv2.boundingRect(cntr)
            box = np.array([x, y, x+w, y+h])
            area = cv2.contourArea(cntr)
            if (area > 1000.0):
                boxes.append(box.tolist())
                cv2.rectangle(mask_img, (x, y), (x+w, y+h), (0, 255, 0), 10)

    # plt.imshow(mask_img)
    # # plt.savefig('predicted.png', dpi=200)
    # plt.show()

    return mask_img, boxes, rects

def read_json(path: pathlib.Path) -> dict:
    with open(str(path), "r") as fp:  # fp is the file pointer
        file = json.loads(s=fp.read())

    return file

def save_json(obj, path: pathlib.Path) -> None:
    with open(path, "w") as fp:  # fp is the file pointer
        json.dump(obj=obj, fp=fp, indent=4, sort_keys=False)


def color_mapping_func(labels: list, mapping: dict):
    """Maps an label (integer or string) to a color"""
    color_list = [mapping[value] for value in labels]
    return color_list


def stats_dataset(
    dataset: Dataset, rcnn_transform: GeneralizedRCNNTransform = False
) -> dict:
    """
    Iterates over the dataset and returns some stats.
    Can be useful to pick the right anchor box sizes.
    """
    stats = {
        "image_height": [],
        "image_width": [],
        "image_mean": [],
        "image_std": [],
        "boxes_height": [],
        "boxes_width": [],
        "boxes_num": [],
        "boxes_area": [],
    }
    for batch in dataset:
        # Batch
        x, y, x_name, y_name = batch["x"], batch["y"], batch["x_name"], batch["y_name"]

        # Transform
        if rcnn_transform:
            x, y = rcnn_transform([x], [y])
            x, y = x.tensors, y[0]

        # Image
        stats["image_height"].append(x.shape[-2])
        stats["image_width"].append(x.shape[-1])
        stats["image_mean"].append(x.mean().item())
        stats["image_std"].append(x.std().item())

        # Target
        wh = box_convert(y["boxes"], "xyxy", "xywh")[:, -2:]
        stats["boxes_height"].append(wh[:, -2])
        stats["boxes_width"].append(wh[:, -1])
        stats["boxes_num"].append(len(wh))
        stats["boxes_area"].append(box_area(y["boxes"]))

    stats["image_height"] = torch.tensor(stats["image_height"], dtype=torch.float)
    stats["image_width"] = torch.tensor(stats["image_width"], dtype=torch.float)
    stats["image_mean"] = torch.tensor(stats["image_mean"], dtype=torch.float)
    stats["image_std"] = torch.tensor(stats["image_std"], dtype=torch.float)
    stats["boxes_height"] = torch.cat(stats["boxes_height"])
    stats["boxes_width"] = torch.cat(stats["boxes_width"])
    stats["boxes_area"] = torch.cat(stats["boxes_area"])
    stats["boxes_num"] = torch.tensor(stats["boxes_num"], dtype=torch.float)

    return stats


def from_file_to_boundingbox(
    file_name: pathlib.Path, groundtruth: bool = True
) -> List[BoundingBox]:
    """Returns a list of BoundingBox objects from groundtruth or prediction."""
    with open(file_name) as json_file:
        file = json.load(json_file)
        labels = file["labels"]
        boxes = file["boxes"]
        scores = file["scores"] if not groundtruth else [None] * len(boxes)

        gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [
        BoundingBox(
            image_name=file_name.stem,
            class_id=l,
            coordinates=tuple(bb),
            format=BBFormat.XYX2Y2,
            bb_type=gt,
            confidence=s,
        )
        for bb, l, s in zip(boxes, labels, scores)
    ]


def from_dict_to_boundingbox(file: dict, name: str, groundtruth: bool = True):
    """Returns list of BoundingBox objects from groundtruth or prediction."""
    labels = file["labels"]
    boxes = file["boxes"]
    scores = np.array(file["scores"].cpu()) if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [
        BoundingBox(
            image_name=name,
            class_id=int(l),
            coordinates=tuple(bb),
            format=BBFormat.XYX2Y2,
            bb_type=gt,
            confidence=s,
        )
        for bb, l, s in zip(boxes, labels, scores)
    ]

def collate_double(batch) -> tuple:
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    """
    x = [sample["x"] for sample in batch]
    y = [sample["y"] for sample in batch]
    x_name = [sample["x_name"] for sample in batch]
    y_name = [sample["y_name"] for sample in batch]
    return default_collate(x), y, x_name, y_name


def collate_single(batch) -> tuple:
    """
    collate function for the ObjectDetectionDataSetSingle.
    Only used by the dataloader.
    """
    x = [sample["x"] for sample in batch]
    x_name = [sample["x_name"] for sample in batch]
    return x, x_name


def roi_metrics(img, gt_boxes, pr_boxes):
    """ compute the precision, recall (coverage) and the F1-score for the given prediction.

    # Arguments
        true_mask: the ground truth mask for the given image
        gt_boxes: the labelled boxes of ground truth regions produced by border following
        pr_boxes: the predicted boxes of skin (or AD) regions

    # Returns
        return the metrics in the following order: recall, precision, f1, acc, dice_coef (which is the same as F1-score), [TP, TN, FP, FN]

    """
    # if len(gt_boxes.shape) == 2:
    #     gt_boxes = np.stack([gt_boxes, gt_boxes], axis=-1)
    #     pr_boxes = np.stack([pr_boxes, pr_boxes], axis=-1)
    #     for boxes in [gt_boxes, pr_boxes]:
    #         for i in range(len(boxes)):
    #             x, y, x2, y2 = boxes[i, :, 0]
    #             boxes[i] = [[x2, y2], [x, y2], [x, y], [x2, y]]

    height = img.shape[0]
    width = img.shape[1]

    # # coordinate transformation
    # for box in gt_boxes:
    #     for point in box:
    #         point[1] = height - point[1]
    # for box in pr_boxes:
    #     for point in box:
    #         point[1] = width - point[1]

    gt_layer = np.zeros([height, width], dtype=np.uint8)
    pr_layer = np.zeros([height, width], dtype=np.uint8)

    gt_boxes = gt_boxes.astype(np.int32)
    pr_boxes = np.array(pr_boxes).astype(np.int32)

    if len(gt_boxes.shape) == 2:
        for box in gt_boxes:
            gt_layer[box[1]:box[3], box[0]:box[2]] = 1
            # plt.imshow(gt_layer)
            # plt.show()
        for box in pr_boxes:
            pr_layer[box[1]:box[3], box[0]:box[2]] = 1
            # plt.imshow(gt_layer)
            # plt.show()
    else:
        for box in gt_boxes:
            cv2.fillPoly(gt_layer, [box.reshape(-1, 1, 2)], 1)
            # plt.imshow(gt_layer)
            # plt.show()
        for box in pr_boxes:
            cv2.fillPoly(pr_layer, [box.reshape(-1, 1, 2)], 1)
            # plt.imshow(gt_layer)
            # plt.show()

    # # iterate over all pixels to find which of them are contained in reference prediction
    # # and which of them are contained in perturbed prediction
    # for i in range(height):
    #     for j in range(width):
    #         # iterate through all the boxes to check if pixel is inside any of them
    #         for box in gt_boxes:
    #             path = pltPath.Path([box[0], box[1], box[2], box[3]])
    #             if path.contains_points([[j, height - i]]):
    #                 # count pixels in both ground truth mask and boxes (TP)
    #                 gt_layer[i, j] = 1.0
    #                 break
    #         for box in pr_boxes:
    #             path = pltPath.Path([box[0], box[1], box[2], box[3]])
    #             if path.contains_points([[j, height - i]]):
    #                 pr_layer[i, j] = 1.0
    #                 break

    gt_area = np.sum(gt_layer)
    pr_area = np.sum(pr_layer)
    TP = np.sum(np.logical_and(pr_layer, gt_layer))
    TN = np.sum(np.logical_and(1-pr_layer, 1-gt_layer))
    FN = gt_area - TP
    FP = pr_area - TP

    # define evaluation metrics
    sen = 0 if gt_area == 0 else TP / gt_area
    pre = 0 if pr_area == 0 else TP / pr_area
    acc = 0 if (TP + TN + FP + FN) == 0 else (TP + TN) / (TP + TN + FP + FN)
    dice = 0 if (TP + FP + TP + FN) == 0 else 2 * TP / (TP + FP + TP + FN)
    iou = 0 if (TP + FP + FN) == 0 else TP / (TP + FP + FN)

    return {'sen': sen, 'pre': pre, 'acc': acc, 'dice': dice, 'iou': iou}