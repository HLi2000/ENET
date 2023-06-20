"""
To test the prediction network and save results
"""

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

from src.utils.pred import convolve_many, rps
from src.utils.utils import TestStore, save_csv

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score

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

    res_pds = []
    res_crop_pds = []
    for k in range(cfg.num_folds):

        log.info(f"Instantiating fold <{k}> \n")
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        cfg.datamodule.k = k
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
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path[k])

        # for predictions use trainer.predict(...)
        # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

        log.info(f"Best ckpt path: {cfg.ckpt_path[k]}")

        outputs = test_store.get_test_outputs()
        res = []
        res_crop = []
        signs = ['cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed']
        if cfg.crop:
            outputs_df = pd.DataFrame.from_records(outputs)
            filenames = outputs_df['filename'].copy()
            filenames = filenames.drop_duplicates()
            for name in filenames:
                outs = outputs_df[outputs_df['filename'] == name].copy().reset_index()
                # outs = outs.loc[[0]]
                x_path = outs['x_path'].get(0)
                target = outs['target'].get(0)
                preds = torch.stack([pred for pred in outs["prob_cat"]]).type(torch.float32)
                preds = torch.mean(preds, axis=0)
                signs_pred = []
                for i, pred in enumerate(preds):
                    preds[i] = pred / torch.sum(pred)
                    signs_pred.append(torch.sum(preds[i] * torch.arange(len(preds[i])).to(pred.device)))

                signs_ae = []
                signs_se = []
                for i, s in enumerate(signs):
                    signs_ae.append(mean_absolute_error(signs_pred[i], target[s]))
                    signs_se.append(mean_squared_error(signs_pred[i], target[s]))

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

                SASSAD_ae = mean_absolute_error(SASSAD_pred, SASSAD)
                SASSAD_se = mean_squared_error(SASSAD_pred, SASSAD)
                # SASSAD_rmse = torch.sqrt(SASSAD_mse)
                TISS_ae = mean_absolute_error(TISS_pred, TISS)
                TISS_se = mean_squared_error(TISS_pred, TISS)
                # TISS_rmse = torch.sqrt(TISS_mse)
                EASI_ae = mean_absolute_error(EASI_pred, EASI)
                EASI_se = mean_squared_error(EASI_pred, EASI)
                # EASI_rmse = torch.sqrt(EASI_mse)

                SASSAD_pmf = torch.zeros_like(SASSAD_pred_pmf)
                SASSAD_pmf[int(SASSAD)] = 1
                SASSAD_rps = rps(SASSAD_pred_pmf, SASSAD_pmf)
                TISS_pmf = torch.zeros_like(TISS_pred_pmf)
                TISS_pmf[int(TISS)] = 1
                TISS_rps = rps(TISS_pred_pmf, TISS_pmf)
                EASI_pmf = torch.zeros_like(EASI_pred_pmf)
                EASI_pmf[int(EASI)] = 1
                EASI_rps = rps(EASI_pred_pmf, EASI_pmf)

            res.append({'filepath': x_path,
                        'cra_pred': signs_pred[0].item(), 'dry_pred': signs_pred[1].item(),
                        'ery_pred': signs_pred[2].item(), 'exc_pred': signs_pred[3].item(),
                        'exu_pred': signs_pred[4].item(), 'lic_pred': signs_pred[5].item(),
                        'oed_pred': signs_pred[6].item(),
                        'cra_ae': signs_ae[0].item(), 'dry_ae': signs_ae[1].item(),
                        'ery_ae': signs_ae[2].item(), 'exc_ae': signs_ae[3].item(),
                        'exu_ae': signs_ae[4].item(), 'lic_ae': signs_ae[5].item(),
                        'oed_ae': signs_ae[6].item(),
                        'cra_se': signs_se[0].item(), 'dry_se': signs_se[1].item(),
                        'ery_se': signs_se[2].item(), 'exc_se': signs_se[3].item(),
                        'exu_se': signs_se[4].item(), 'lic_se': signs_se[5].item(),
                        'oed_se': signs_se[6].item(),
                        'SASSAD': SASSAD.item(), 'SASSAD_pred': SASSAD_pred.item(),
                        'TISS': TISS.item(), 'TISS_pred': TISS_pred.item(),
                        'EASI': EASI.item(), 'EASI_pred': EASI_pred.item(),
                        'SASSAD_ae': SASSAD_ae.item(), 'SASSAD_se': SASSAD_se.item(),
                        'SASSAD_rps': SASSAD_rps.item(),
                        'TISS_ae': TISS_ae.item(), 'TISS_se': TISS_se.item(),
                        'TISS_rps': TISS_rps.item(),
                        'EASI_ae': EASI_ae.item(), 'EASI_se': EASI_se.item(),
                        'EASI_rps': EASI_rps.item(),
                        'k': k,
                        })

            for out in outputs:
                x_path = out['x_path']
                target = out['target']
                preds = out["prob_cat"].clone().cpu()
                signs_pred = []
                for i, pred in enumerate(preds):
                    preds[i] = pred / torch.sum(pred)
                    signs_pred.append(torch.sum(preds[i] * torch.arange(len(preds[i])).to(pred.device)))

                signs_ae = []
                signs_se = []
                for i, s in enumerate(signs):
                    signs_ae.append(mean_absolute_error(signs_pred[i], target[s]))
                    signs_se.append(mean_squared_error(signs_pred[i], target[s]))

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

                SASSAD_ae = mean_absolute_error(SASSAD_pred, SASSAD)
                SASSAD_se = mean_squared_error(SASSAD_pred, SASSAD)
                # SASSAD_rmse = torch.sqrt(SASSAD_mse)
                TISS_ae = mean_absolute_error(TISS_pred, TISS)
                TISS_se = mean_squared_error(TISS_pred, TISS)
                # TISS_rmse = torch.sqrt(TISS_mse)
                EASI_ae = mean_absolute_error(EASI_pred, EASI)
                EASI_se = mean_squared_error(EASI_pred, EASI)
                # EASI_rmse = torch.sqrt(EASI_mse)

                SASSAD_pmf = torch.zeros_like(SASSAD_pred_pmf)
                SASSAD_pmf[int(SASSAD)] = 1
                SASSAD_rps = rps(SASSAD_pred_pmf, SASSAD_pmf)
                TISS_pmf = torch.zeros_like(TISS_pred_pmf)
                TISS_pmf[int(TISS)] = 1
                TISS_rps = rps(TISS_pred_pmf, TISS_pmf)
                EASI_pmf = torch.zeros_like(EASI_pred_pmf)
                EASI_pmf[int(EASI)] = 1
                EASI_rps = rps(EASI_pred_pmf, EASI_pmf)

                res_crop.append({'filepath': x_path,
                                 'cra_pred': signs_pred[0].item(), 'dry_pred': signs_pred[1].item(),
                                 'ery_pred': signs_pred[2].item(), 'exc_pred': signs_pred[3].item(),
                                 'exu_pred': signs_pred[4].item(), 'lic_pred': signs_pred[5].item(),
                                 'oed_pred': signs_pred[6].item(),
                                 'cra_ae': signs_ae[0].item(), 'dry_ae': signs_ae[1].item(),
                                 'ery_ae': signs_ae[2].item(), 'exc_ae': signs_ae[3].item(),
                                 'exu_ae': signs_ae[4].item(), 'lic_ae': signs_ae[5].item(),
                                 'oed_ae': signs_ae[6].item(),
                                 'cra_se': signs_se[0].item(), 'dry_se': signs_se[1].item(),
                                 'ery_se': signs_se[2].item(), 'exc_se': signs_se[3].item(),
                                 'exu_se': signs_se[4].item(), 'lic_se': signs_se[5].item(),
                                 'oed_se': signs_se[6].item(),
                                 'SASSAD': SASSAD.item(), 'SASSAD_pred': SASSAD_pred.item(),
                                 'TISS': TISS.item(), 'TISS_pred': TISS_pred.item(),
                                 'EASI': EASI.item(), 'EASI_pred': EASI_pred.item(),
                                 'SASSAD_ae': SASSAD_ae.item(), 'SASSAD_se': SASSAD_se.item(),
                                 'SASSAD_rps': SASSAD_rps.item(),
                                 'TISS_ae': TISS_ae.item(), 'TISS_se': TISS_se.item(),
                                 'TISS_rps': TISS_rps.item(),
                                 'EASI_ae': EASI_ae.item(), 'EASI_se': EASI_se.item(),
                                 'EASI_rps': EASI_rps.item(),
                                 'k': k,
                                 })

            res_pds.append(pd.DataFrame.from_records(res, index='filepath'))
            res_crop_pds.append(pd.DataFrame.from_records(res_crop, index='filepath'))

        else:
            for out in outputs:
                x_path = out['x_path']
                target = out['target']
                preds = out["prob_cat"].clone().cpu()
                signs_pred = []
                for i, pred in enumerate(preds):
                    preds[i] = pred / torch.sum(pred)
                    signs_pred.append(torch.sum(preds[i] * torch.arange(len(preds[i])).to(pred.device)))

                signs_ae = []
                signs_se = []
                for i, s in enumerate(signs):
                    signs_ae.append(mean_absolute_error(signs_pred[i], target[s]))
                    signs_se.append(mean_squared_error(signs_pred[i], target[s]))

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

                SASSAD_ae = mean_absolute_error(SASSAD_pred, SASSAD)
                SASSAD_se = mean_squared_error(SASSAD_pred, SASSAD)
                # SASSAD_rmse = torch.sqrt(SASSAD_mse)
                TISS_ae = mean_absolute_error(TISS_pred, TISS)
                TISS_se = mean_squared_error(TISS_pred, TISS)
                # TISS_rmse = torch.sqrt(TISS_mse)
                EASI_ae = mean_absolute_error(EASI_pred, EASI)
                EASI_se = mean_squared_error(EASI_pred, EASI)
                # EASI_rmse = torch.sqrt(EASI_mse)

                SASSAD_pmf = torch.zeros_like(SASSAD_pred_pmf)
                SASSAD_pmf[int(SASSAD)] = 1
                SASSAD_rps = rps(SASSAD_pred_pmf, SASSAD_pmf)
                TISS_pmf = torch.zeros_like(TISS_pred_pmf)
                TISS_pmf[int(TISS)] = 1
                TISS_rps = rps(TISS_pred_pmf, TISS_pmf)
                EASI_pmf = torch.zeros_like(EASI_pred_pmf)
                EASI_pmf[int(EASI)] = 1
                EASI_rps = rps(EASI_pred_pmf, EASI_pmf)

                res.append({'filepath': x_path,
                            'cra_pred': signs_pred[0].item(), 'dry_pred': signs_pred[1].item(),
                            'ery_pred': signs_pred[2].item(), 'exc_pred': signs_pred[3].item(),
                            'exu_pred': signs_pred[4].item(), 'lic_pred': signs_pred[5].item(),
                            'oed_pred': signs_pred[6].item(),
                            'cra_ae': signs_ae[0].item(), 'dry_ae': signs_ae[1].item(),
                            'ery_ae': signs_ae[2].item(), 'exc_ae': signs_ae[3].item(),
                            'exu_ae': signs_ae[4].item(), 'lic_ae': signs_ae[5].item(),
                            'oed_ae': signs_ae[6].item(),
                            'cra_se': signs_se[0].item(), 'dry_se': signs_se[1].item(),
                            'ery_se': signs_se[2].item(), 'exc_se': signs_se[3].item(),
                            'exu_se': signs_se[4].item(), 'lic_se': signs_se[5].item(),
                            'oed_se': signs_se[6].item(),
                            'SASSAD': SASSAD.item(), 'SASSAD_pred': SASSAD_pred.item(),
                            'TISS': TISS.item(), 'TISS_pred': TISS_pred.item(),
                            'EASI': EASI.item(), 'EASI_pred': EASI_pred.item(),
                            'SASSAD_ae': SASSAD_ae.item(), 'SASSAD_se': SASSAD_se.item(),
                            'SASSAD_rps': SASSAD_rps.item(),
                            'TISS_ae': TISS_ae.item(), 'TISS_se': TISS_se.item(),
                            'TISS_rps': TISS_rps.item(),
                            'EASI_ae': EASI_ae.item(), 'EASI_se': EASI_se.item(),
                            'EASI_rps': EASI_rps.item(),
                            'k': k,
                            })

            res_pds.append(pd.DataFrame.from_records(res, index='filepath'))

    data_dir = pathlib.Path(cfg.paths.data_dir) / cfg.dataset

    print(len(res_pds))
    if cfg.crop:
        res_pd = pd.concat(res_pds)
        res_crop_pd = pd.concat(res_crop_pds)
        pred_csv_dir = data_dir / f"metadata_{cfg.seg_type}_{cfg.method}_pred_{cfg.perturbation}.csv"
        pred_csv = pd.read_csv(pred_csv_dir, index_col='filepath')
        auto_csv_dir = data_dir / f"metadata_{cfg.seg_type}_{cfg.method}_crop_{cfg.perturbation}.csv"
        auto_csv = pd.read_csv(auto_csv_dir, index_col='filepath')
        auto_csv = pd.concat([auto_csv, pred_csv])
        # auto_csv_dir = data_dir / f"metadata_{cfg.seg_type}_man_crop_{cfg.perturbation}.csv"
    else:
        if cfg.method == 'seg':
            res_pd = pd.concat(res_pds)
            pred_csv_dir = data_dir / f"metadata_{cfg.seg_type}_{cfg.method}_pred_whole_{cfg.perturbation}.csv"
            pred_csv = pd.read_csv(pred_csv_dir, index_col='filepath')
            auto_csv_dir = data_dir / f"metadata_{cfg.seg_type}_{cfg.method}_whole_{cfg.perturbation}.csv"
            auto_csv = pd.read_csv(auto_csv_dir, index_col='filepath')
            auto_csv = pd.concat([auto_csv, pred_csv])
        else:
            res_pd = pd.concat(res_pds)
            auto_csv_dir = data_dir / f"metadata.csv"
            auto_csv = pd.read_csv(auto_csv_dir, index_col='filepath')

    res_csv = auto_csv.join(res_pd, how="inner")
    res_csv = res_csv.reset_index(level=0).dropna()
    if cfg.crop:
        csv_name = data_dir / f"pred_{cfg.seg_type}_{cfg.method}_crop_{cfg.net}_{cfg.perturbation}.csv"
    else:
        if cfg.method == 'seg':
            csv_name = data_dir / f"pred_{cfg.seg_type}_{cfg.method}_{cfg.net}_{cfg.perturbation}_whole.csv"
        else:
            csv_name = data_dir / f"pred_{cfg.seg_type}_{cfg.method}_{cfg.net}_{cfg.perturbation}_raw.csv"
    save_csv(res_csv, csv_name, 'x')

    if cfg.crop:
        res_crop_csv = auto_csv.join(res_crop_pd, how="inner")
        res_crop_csv = res_crop_csv.reset_index(level=0).dropna()
        csv_crop_name = data_dir / f"pred_{cfg.seg_type}_{cfg.method}_crop_{cfg.net}_{cfg.perturbation}_crops.csv"
        save_csv(res_crop_csv, csv_crop_name, 'x')


    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
