import pathlib
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import pandas as pd
import torch

from src.utils.pred import convolve_many
from src.utils.utils import TestStore


from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from torchmetrics.functional import mean_absolute_error, mean_squared_error

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    class_weights = datamodule.class_weights

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, class_weights=class_weights)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    test_store = TestStore()
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=[test_store])

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    log.info(f"Best ckpt path: {cfg.ckpt_path}")

    outputs = test_store.get_test_outputs()
    res = []
    if cfg.crop:
        outputs_df = pd.DataFrame.from_records(outputs)
        filenames = outputs_df['filename'].copy()
        filenames = filenames.drop_duplicates()
        for name in filenames:
            outs = outputs_df[outputs_df['filename'] == name].copy().reset_index()
            x_path = outs['x_path'].get(0)
            target = outs['target'].get(0)
            preds = torch.stack([pred for pred in outs["prob_cat"]]).type(torch.float32)
            preds = torch.mean(preds, axis=0)

            SASSAD = target['cra'] + target['dry'] + target['ery'] + \
                     target['exc'] + target['exu'] + target['lic']
            TISS = target['ery'] + target['exc'] + target['oed']
            EASI = target['ery'] + target['exc'] + target['lic'] + target['oed']

            SASSAD_pred_pmf = convolve_many([preds[0], preds[1], preds[2], preds[3], preds[4], preds[5]])
            SASSAD_pred = torch.sum(SASSAD_pred_pmf * torch.arange(len(SASSAD_pred_pmf)))
            TISS_pred_pmf = convolve_many([preds[2], preds[3], preds[6]])
            TISS_pred = torch.sum(TISS_pred_pmf * torch.arange(len(TISS_pred_pmf)))
            EASI_pred_pmf = convolve_many([preds[2], preds[3], preds[5], preds[6]])
            EASI_pred = torch.sum(EASI_pred_pmf * torch.arange(len(EASI_pred_pmf)))

            SASSAD_mae = mean_absolute_error(SASSAD_pred, SASSAD)
            SASSAD_mse = mean_squared_error(SASSAD_pred, SASSAD)
            SASSAD_rmse = torch.sqrt(SASSAD_mse)
            TISS_mae = mean_absolute_error(TISS_pred, TISS)
            TISS_mse = mean_squared_error(TISS_pred, TISS)
            TISS_rmse = torch.sqrt(TISS_mse)
            EASI_mae = mean_absolute_error(EASI_pred, EASI)
            EASI_mse = mean_squared_error(EASI_pred, EASI)
            EASI_rmse = torch.sqrt(EASI_mse)

            res.append({'filepath': x_path,
                        'SASSAD': SASSAD.item(), 'SASSAD_pred': SASSAD_pred.item(),
                        'TISS': TISS.item(), 'TISS_pred': TISS_pred.item(),
                        'EASI': EASI.item(), 'EASI_pred': EASI_pred.item(),
                        'SASSAD_mae': SASSAD_mae.item(), 'SASSAD_mse': SASSAD_mse.item(), 'SASSAD_rmse': SASSAD_rmse.item(),
                        'TISS_mae': TISS_mae.item(), 'TISS_mse': TISS_mse.item(), 'TISS_rmse': TISS_rmse.item(),
                        'EASI_mae': EASI_mae.item(), 'EASI_mse': EASI_mse.item(), 'EASI_rmse': EASI_rmse.item(),
                        })

    res_pd = pd.DataFrame.from_records(res, index='filepath')

    data_dir = pathlib.Path(cfg.paths.data_dir) / cfg.dataset
    if cfg.crop:
        auto_csv_dir = data_dir / f"metadata_{cfg.seg_type}_pred_{cfg.perturbation}.csv"
        auto_csv = pd.read_csv(auto_csv_dir, index_col='filepath')

        res_csv = auto_csv.join(res_pd, how="inner")
        res_csv = res_csv.reset_index(level=0).dropna()
        csv_name = data_dir / f"pred_{cfg.seg_type}_auto_crop_{cfg.perturbation}.csv"
        res_csv.to_csv(csv_name, encoding='utf-8', index=False)


    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
