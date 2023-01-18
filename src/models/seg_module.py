from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, Dice
import segmentation_models_pytorch as smp

from itertools import chain

from torchmetrics.classification import BinaryJaccardIndex

from src.metrics.enumerators import MethodAveragePrecision

from src.models.components.loss import DiceLoss, FocalLoss
from src.utils.roi import from_dict_to_boundingbox, roi_metrics, draw_box_countours


class SegModule(LightningModule):
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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        monitor: str = None,
        auto_crop: bool = False,
        encoder_name: str = "efficientnet_b3",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Model
        self.model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights="imagenet",
            classes=1,
            activation='sigmoid'
        )

        self.criterion0 = DiceLoss()
        self.criterion1 = FocalLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()
        self.test_iou = BinaryJaccardIndex()

        self.train_dice = Dice()
        self.val_dice = Dice()
        self.test_dice = Dice()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_iou_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_iou_best.reset()

    def step(self, batch: Any):
        x, y, x_name = batch

        logits = self.forward(x)

        loss = self.criterion0(logits, y) + self.criterion1(logits, y)

        preds = (logits > 0.5).float()

        return loss, preds, y.type(torch.uint8)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_iou(preds.flatten(), targets.flatten())
        self.train_dice(preds.flatten(), targets.flatten())
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/IoU", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/Dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_iou(preds.flatten(), targets.flatten())
        self.val_dice(preds.flatten(), targets.flatten())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/IoU", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/Dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        pass


    def test_step(self, batch: Any, batch_idx: int):
        x, y, x_name = batch

        loss, preds, targets = self.step(batch)

        self.test_iou(preds.flatten(), targets.flatten())
        self.test_dice(preds.flatten(), targets.flatten())
        self.log("test/IoU", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/Dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)

        x = x * torch.concat([y, y, y], dim=1)

        _, boxes_y, rects_y = draw_box_countours(y[0, 0].cpu().numpy(), rotated=True)
        _, boxes_p, rects_p = draw_box_countours(preds[0, 0].cpu().numpy(), rotated=True)
        roi_metric = roi_metrics(x[0,0].cpu().numpy(), boxes_y, boxes_p)
        sen, pre, acc, dice, iou = roi_metric['sen'], roi_metric['pre'], roi_metric['acc'], roi_metric['dice'], roi_metric['iou']
        self.log("test/sen", sen, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/pre", pre, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "x_name": x_name[0], "x": x[0].cpu().numpy(), "pred_boxes": boxes_p, "pred_rects": rects_p,
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
