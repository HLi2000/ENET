"""
Module that import and train/test the model with chosen metrics
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from collections import OrderedDict
from itertools import chain

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from src.models.components.backbone_resnet import (
    BackboneWithFPN,
    ResNetBackbones,
    get_resnet_backbone,
    get_resnet_fpn_backbone,
)
from src.metrics.enumerators import MethodAveragePrecision
from src.metrics.pascal_voc_evaluator import (
    get_pascalvoc_metrics,
)
from src.utils.roi import from_dict_to_boundingbox, roi_metrics
from ast import literal_eval as make_tuple
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def get_anchor_generator(
    anchor_size: Optional[Tuple[Tuple[int]]] = None,
    aspect_ratios: Optional[Tuple[Tuple[float]]] = None,
) -> AnchorGenerator:
    """Returns the anchor generator."""
    if anchor_size is None:
        anchor_size = ((16,), (32,), (64,), (128,))
    if aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_size)

    anchor_generator = AnchorGenerator(sizes=anchor_size, aspect_ratios=aspect_ratios)
    return anchor_generator


def get_roi_pool(
    featmap_names: Optional[List[str]] = None,
    output_size: int = 7,
    sampling_ratio: int = 2,
) -> MultiScaleRoIAlign:
    """Returns the ROI Pooling"""
    if featmap_names is None:
        # default for resnet with FPN
        featmap_names = ["0", "1", "2", "3"]

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=output_size,
        sampling_ratio=sampling_ratio,
    )

    return roi_pooler


def get_faster_rcnn(
    backbone: torch.nn.Module,
    anchor_generator: AnchorGenerator,
    roi_pooler: MultiScaleRoIAlign,
    num_classes: int,
    image_mean: List[float] = [0.485, 0.456, 0.406],
    image_std: List[float] = [0.229, 0.224, 0.225],
    min_size: int = 512,
    max_size: int = 1024,
    **kwargs,
) -> FasterRCNN:
    """Returns the Faster-RCNN model. Default normalization: ImageNet"""
    model = FasterRCNN(
        backbone=backbone,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        num_classes=num_classes,
        image_mean=image_mean,  # ImageNet
        image_std=image_std,  # ImageNet
        min_size=min_size,
        max_size=max_size,
        **kwargs,
    )
    model.num_classes = num_classes
    model.image_mean = image_mean
    model.image_std = image_std
    model.min_size = min_size
    model.max_size = max_size

    return model


def get_faster_rcnn_resnet(
    num_classes: int,
    backbone_name: ResNetBackbones,
    anchor_size: Tuple[Tuple[int, ...], ...],
    aspect_ratios: Tuple[Tuple[float, ...]],
    fpn: bool = True,
    min_size: int = 512,
    max_size: int = 1024,
    **kwargs,
) -> FasterRCNN:
    """
    Returns the Faster-RCNN model with resnet backbone with and without fpn.
    anchor_size can be for example: ((16,), (32,), (64,), (128,))
    aspect_ratios can be for example: ((0.5, 1.0, 2.0),)
    Please note that you specify the aspect ratios for all layers, because we perform:
    aspect_ratios = aspect_ratios * len(anchor_size)
    If you wish more control, change this line accordingly.
    """

    # Backbone
    if fpn:
        backbone: BackboneWithFPN = get_resnet_fpn_backbone(backbone_name=backbone_name)
    else:
        backbone: torch.nn.Sequential = get_resnet_backbone(backbone_name=backbone_name)

    # Anchors
    anchor_size = make_tuple(anchor_size)
    aspect_ratios = make_tuple(aspect_ratios)

    aspect_ratios = aspect_ratios * len(anchor_size)
    # print(anchor_size)
    # print(aspect_ratios)
    anchor_generator = get_anchor_generator(
        anchor_size=anchor_size, aspect_ratios=aspect_ratios
    )

    # ROI Pool
    # performing a forward pass to get the number of featuremap names
    # this is required for the get_roi_pool function
    # TODO: there is probably a better way to get the featuremap names (without a forward pass)
    with torch.no_grad():
        backbone.eval()
        random_input = torch.rand(size=(1, 3, 512, 512))
        features = backbone(random_input)

    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    featmap_names = [key for key in features.keys() if key.isnumeric()]

    roi_pool = get_roi_pool(featmap_names=featmap_names)

    # Model
    return get_faster_rcnn(
        backbone=backbone,
        anchor_generator=anchor_generator,
        roi_pooler=roi_pool,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        **kwargs,
    )

class FasterRCNNModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        monitor: str = None,
        auto_crop: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model"])

        # Model
        self.model = model

        # metric objects for calculating and averaging accuracy across batches
        self.val_map = get_pascalvoc_metrics
        self.test_map = MeanAveragePrecision()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_map_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_map_best.reset()

    def step(self, batch: Any):
        pass

    def training_step(self, batch: Any, batch_idx: int):
        # Batch
        x, y, x_name, y_name = batch

        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values()) # classification + regression

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        # Batch
        x, y, x_name, y_name = batch

        # Inference
        preds = self.model(x)

        gt_boxes = [
            from_dict_to_boundingbox(file=target, name=name, groundtruth=True)
            for target, name in zip(y, x_name)
        ]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [
            from_dict_to_boundingbox(file=pred, name=name, groundtruth=False)
            for pred, name in zip(preds, x_name)
        ]
        pred_boxes = list(chain(*pred_boxes))

        metric = self.val_map(
            gt_boxes=gt_boxes,
            det_boxes=pred_boxes,
            iou_threshold=0.5,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=True,
        )

        map_50 = metric["m_ap"]

        metric = self.val_map(
            gt_boxes=gt_boxes,
            det_boxes=pred_boxes,
            iou_threshold=0.75,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=True,
        )

        map_75 = metric["m_ap"]
        map = (map_50+map_75)/2

        # self.val_map.update(target=y, preds=preds)
        # metric = self.val_map.compute()
        #
        # map, map_50, map_75 = metric["map"], metric["map_50"], metric["map_75"]

        self.log("val/mAP", map, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mAP_50", map_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mAP_75", map_75, on_step=False, on_epoch=True, prog_bar=True)

        self.val_map_best(map)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mAP_best", self.val_map_best.compute(), prog_bar=True)

        sen, pre, acc, dice, iou = [], [], [], [], []
        for i in range(len(x)):
            prs = [box.cpu().numpy() for id, box in enumerate(preds[i]["boxes"]) if preds[i]["scores"][id] > 0.5]
            roi_metric = roi_metrics(x[i][0].cpu().numpy(), y[i]["boxes"].cpu().numpy(), prs)
            sen.append(roi_metric['sen'])
            pre.append(roi_metric['pre'])
            acc.append(roi_metric['acc'])
            dice.append(roi_metric['dice'])
            iou.append(roi_metric['iou'])
        sen, pre, acc, dice, iou = np.mean(sen), np.mean(pre), np.mean(acc), np.mean(dice), np.mean(iou)
        self.log("val/sen", sen, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/pre", pre, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        return {"pred_boxes": preds, "gt_boxes": y}

    def validation_epoch_end(self, outputs: List[Any]):
        pass


    def test_step(self, batch: Any, batch_idx: int):
        # Batch
        x, y, x_name, y_name = batch

        # Inference
        preds = self.model(x)

        if self.hparams.auto_crop:
            map, map_50, map_75 = torch.tensor(-1), torch.tensor(-1), torch.tensor(-1)
        else:
            gt_boxes = [
                from_dict_to_boundingbox(file=target, name=name, groundtruth=True)
                for target, name in zip(y, x_name)
            ]
            gt_boxes = list(chain(*gt_boxes))

            pred_boxes = [
                from_dict_to_boundingbox(file=pred, name=name, groundtruth=False)
                for pred, name in zip(preds, x_name)
            ]
            pred_boxes = list(chain(*pred_boxes))

            metric = self.val_map(
                gt_boxes=gt_boxes,
                det_boxes=pred_boxes,
                iou_threshold=0.5,
                method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                generate_table=True,
            )

            map_50 = metric["m_ap"]

            metric = self.val_map(
                gt_boxes=gt_boxes,
                det_boxes=pred_boxes,
                iou_threshold=0.75,
                method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                generate_table=True,
            )

            map_75 = metric["m_ap"]
            map = (map_50 + map_75) / 2

            # self.test_map.update(target=y, preds=preds)
            # metric = self.test_map.compute()
            #
            # map, map_50, map_75 = metric["map"], metric["map_50"], metric["map_75"]
            self.log("test/mAP", map, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAP_50", map_50, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAP_75", map_75, on_step=False, on_epoch=True, prog_bar=True)

        sen, pre, acc, dice, iou = [], [], [], [], []
        for i in range(len(x)):
            prs = [box.cpu().numpy() for id, box in enumerate(preds[i]["boxes"]) if preds[i]["scores"][id] > 0.5]
            roi_metric = roi_metrics(x[i][0].cpu().numpy(), y[i]["boxes"].cpu().numpy(), prs)
            sen.append(roi_metric['sen'])
            pre.append(roi_metric['pre'])
            acc.append(roi_metric['acc'])
            dice.append(roi_metric['dice'])
            iou.append(roi_metric['iou'])
        sen, pre, acc, dice, iou = np.mean(sen), np.mean(pre), np.mean(acc), np.mean(dice), np.mean(iou)
        self.log("test/sen", sen, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/pre", pre, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "x_name": x_name[0], "x": x[0].cpu().numpy(), "pred_boxes": preds[0],
            "mAP": map.item(), "mAP_50": map_50.item(), "mAP_75": map_75.item(),
            "sen": sen, "pre": pre, "acc": acc, "dice": dice, "iou": iou,
        }

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            if self.hparams.monitor is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.hparams.monitor,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            elif self.hparams.monitor is None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            else:
                raise ValueError("Wrong settings of lr_scheduler!")
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "roi.yaml")
    _ = hydra.utils.instantiate(cfg)
