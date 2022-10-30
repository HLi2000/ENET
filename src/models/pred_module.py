import pathlib
from typing import Any, List

import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MeanAbsoluteError, MeanSquaredError
from torchmetrics.functional import mean_absolute_error, mean_squared_error

from src.utils.pred import ordinariser, ordinariser_reversed, proba_ordinal_to_categorical, convolve_many


class PredModule(LightningModule):
    """Example of LightningModule for Prediction.

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
        ordinal: bool = True,
        n_classes: int = 4,
        crop: bool = True,
        class_weights = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model"])

        self.model = model

        # loss function
        if class_weights is not None:
            self.criterion = torch.nn.BCELoss(weight=class_weights)
        else:
            self.criterion = torch.nn.BCELoss()
        self.activation = torch.nn.Sigmoid()

        # Set up attributes for computing the MAE
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mae_best = MinMetric()
        self.val_mse_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_mae_best.reset()
        self.val_mse_best.reset()

    def step(self, batch: Any):
        x, y_batch, x_path, _ = batch
        outputs = self.forward(x)
        logits = self.activation(outputs)
        if self.hparams.ordinal:
            y_ordinal = []
            targets = []
            for y in y_batch:
                y_tensor = torch.tensor([value for key, value in y.items()])
                targets.append(y_tensor)
                y_ordinal.append(ordinariser(y_tensor, self.hparams.n_classes))
            y_ordinal = torch.stack(y_ordinal, axis=0).type(torch.float32).to(logits.device)
            targets = torch.stack(targets, axis=0).type(torch.float32).to(logits.device)
            loss = self.criterion(logits, y_ordinal)

            prob_cat = []
            for i in range(logits.size()[0]): #batch
                prob_cat.append(proba_ordinal_to_categorical(logits[i]))
            prob_cat = torch.stack(prob_cat, axis=0).to(logits.device)
            # preds = torch.argmax(prob_cat, dim=2)
            preds = torch.tensor([
                [torch.sum(p * torch.arange(len(p)).to(logits.device)) for p in prob]
                                 for prob in prob_cat]).to(logits.device)
        else:
            loss = self.criterion(logits, y_batch)
            preds = torch.argmax(logits, dim=1)
        return loss, preds, targets, prob_cat

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.train_mse(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/MAE", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/MSE", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_mae(preds, targets)
        self.val_mse(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/MAE", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/MSE", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        mae = self.val_mae.compute()  # get current val acc
        self.val_mae_best(mae)  # update best so far val acc
        mse = self.val_mse.compute()  # get current val acc
        self.val_mse_best(mse)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/MAE_best", self.val_mae_best.compute(), prog_bar=True)
        self.log("val/MSE_best", self.val_mse_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        _, y, x_path, _ = batch
        loss, preds, targets, prob_cat = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_mae(preds, targets)
        self.test_mse(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/MAE", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/MSE", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)

        filename = pathlib.Path(x_path[0]).name
        filename = "_".join(filename.split('_')[:-1])
        return {'filename': filename, 'x_path': x_path[0], "prob_cat": prob_cat[0], "target": y[0]}

    def test_epoch_end(self, outputs: List[Any]):
        pass
        # res = []
        # if self.hparams.crop:
        #     outputs_df = pd.DataFrame.from_records(outputs)
        #     filenames = outputs_df['filename'].copy()
        #     filenames = filenames.drop_duplicates()
        #     for name in filenames:
        #         outs = outputs_df[outputs_df['filename'] == name].copy().reset_index()
        #         x_path = outs['x_path'].get(0)
        #         target = outs['target'].get(0)
        #         preds = torch.stack([pred for pred in outs["prob_cat"]]).type(torch.float32)
        #         preds = torch.mean(preds, axis=0)
        #
        #         SASSAD = target['cra'] + target['dry'] + target['ery'] + \
        #                  target['exc'] + target['exu'] + target['lic']
        #         TISS = target['ery'] + target['exc'] + target['oed']
        #         EASI = target['ery'] + target['exc'] + target['lic'] + target['oed']
        #
        #         SASSAD_pred_pmf = convolve_many([preds[0],preds[1],preds[2],preds[3],preds[4],preds[5]])
        #         SASSAD_pred = torch.sum(SASSAD_pred_pmf * torch.arange(len(SASSAD_pred_pmf)))
        #         TISS_pred_pmf = convolve_many([preds[2], preds[3], preds[6]])
        #         TISS_pred = torch.sum(TISS_pred_pmf * torch.arange(len(TISS_pred_pmf)))
        #         EASI_pred_pmf = convolve_many([preds[2], preds[3], preds[5], preds[6]])
        #         EASI_pred = torch.sum(EASI_pred_pmf * torch.arange(len(EASI_pred_pmf)))
        #
        #         SASSAD_mae = mean_absolute_error(SASSAD_pred, SASSAD)
        #         SASSAD_mse = mean_squared_error(SASSAD_pred, SASSAD)
        #         SASSAD_rmse = torch.sqrt(SASSAD_mse)
        #         TISS_mae = mean_absolute_error(TISS_pred, TISS)
        #         TISS_mse = mean_squared_error(TISS_pred, TISS)
        #         TISS_rmse = torch.sqrt(TISS_mse)
        #         EASI_mae = mean_absolute_error(EASI_pred, EASI)
        #         EASI_mse = mean_squared_error(EASI_pred, EASI)
        #         EASI_rmse = torch.sqrt(EASI_mse)
        #
        #         res.append({'crop': x_path,
        #                     'SASSAD_mae': SASSAD_mae, 'SASSAD_mse': SASSAD_mse, 'SASSAD_rmse': SASSAD_rmse,
        #                     'TISS_mae': TISS_mae, 'TISS_mse': TISS_mse, 'TISS_rmse': TISS_rmse,
        #                     'EASI_mae': EASI_mae, 'EASI_mse': EASI_mse, 'EASI_rmse': EASI_rmse,
        #                     })
        #
        # return res

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
