"""
To train and test the segmentation network and save crops/segmentations and corresponding metadata
"""

import numpy as np
import pyrootutils
import pathlib

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from matplotlib import pyplot as plt

from src.datamodules.components.transfroms import re_normalize
from src.utils.roi import crop_square
from src.utils.utils import TestStore

from typing import List, Optional, Tuple

import pandas as pd
import csv

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    test_store = TestStore()
    callbacks.append(test_store)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    res_dicts = test_store.get_test_outputs()

    score_dir = pathlib.Path(cfg.data_dir, cfg.dataset, 'metadata.csv')
    score_file = pd.read_csv(score_dir)

    if cfg.auto_crop:
        csv_dir0 = pathlib.Path(cfg.data_dir, cfg.dataset, f'metadata_{cfg.seg_type}_seg_whole_{cfg.perturbation}.csv')
        csv_file0 = open(csv_dir0, 'w', newline='')
        writer0 = csv.writer(csv_file0)
        writer0.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                          'filename', 'crop', 'filepath', 'task'])

        csv_dir = pathlib.Path(cfg.data_dir, cfg.dataset, f'metadata_{cfg.seg_type}_seg_crop_{cfg.perturbation}.csv')
        csv_file = open(csv_dir, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                         'filename', 'crop', 'filepath', 'task'])

        length = len(res_dicts)
        for i, res in enumerate(res_dicts):
            x_name, x, boxes, rects = res['x_name'], res['x'], res['pred_boxes'], res['pred_rects']

            index = pd.Index(score_file['filename']).get_loc(x_name)
            task = score_file['task'].get(index)

            img = np.moveaxis(re_normalize(x), source=0, destination=-1)

            img_dir = pathlib.Path(cfg.data_dir, cfg.dataset, task, f"seg_{cfg.seg_type}_whole_{cfg.perturbation}")
            img_dir.mkdir(parents=True, exist_ok=True)
            img_file_name = x_name.split(".")[0] + "_whole" + ".jpg"
            img_file_dir = pathlib.Path(img_dir, img_file_name)
            plt.imsave(img_file_dir, img)

            writer0.writerow(
                [score_file['refno'].get(index), score_file['visno'].get(index),
                 score_file['ethnic'].get(index),
                 int(score_file['cra'].get(index)),
                 int(score_file['dry'].get(index)),
                 int(score_file['ery'].get(index)),
                 int(score_file['exc'].get(index)),
                 int(score_file['exu'].get(index)),
                 int(score_file['lic'].get(index)),
                 int(score_file['oed'].get(index)),
                 x_name, img_file_name, img_file_dir, task,
                 ])

            print(f'{task.upper()}: {i}/{length} {img_file_name} created')

            crop_dir = pathlib.Path(cfg.data_dir, cfg.dataset, task, f"seg_{cfg.seg_type}_crops_{cfg.perturbation}")
            crop_dir.mkdir(parents=True, exist_ok=True)

            no = 0
            for id, box in enumerate(boxes):
                crops = crop_square(img, box, rects[id])
                for crop in crops:
                    crop_file_name = x_name.split(".")[0] + "_crop" + str(no) + ".jpg"
                    crop_file_dir = pathlib.Path(crop_dir, crop_file_name)
                    plt.imsave(crop_file_dir, crop)

                    writer.writerow(
                        [score_file['refno'].get(index), score_file['visno'].get(index),
                         score_file['ethnic'].get(index),
                         int(score_file['cra'].get(index)),
                         int(score_file['dry'].get(index)),
                         int(score_file['ery'].get(index)),
                         int(score_file['exc'].get(index)),
                         int(score_file['exu'].get(index)),
                         int(score_file['lic'].get(index)),
                         int(score_file['oed'].get(index)),
                         x_name, crop_file_name, crop_file_dir, task,
                         ])

                    no += 1
                    print(f'{task.upper()}: {i}/{length} {crop_file_name} created')

        csv_file.close()
    else:
        csv_dir0 = pathlib.Path(cfg.data_dir, cfg.dataset,
                                f'metadata_{cfg.seg_type}_seg_pred_whole_{cfg.perturbation}.csv')
        csv_file0 = open(csv_dir0, 'w', newline='')
        writer0 = csv.writer(csv_file0)
        writer0.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                          'filename', 'crop', 'filepath', 'task'])

        csv_dir1 = pathlib.Path(cfg.data_dir, cfg.dataset, f'metadata_{cfg.seg_type}_seg_pred_{cfg.perturbation}.csv')
        csv_file1 = open(csv_dir1, 'w', newline='')
        writer1 = csv.writer(csv_file1)
        writer1.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                          'filename', 'crop', 'filepath', 'task'])

        csv_dir2 = pathlib.Path(cfg.data_dir, cfg.dataset, f'results_{cfg.seg_type}_seg_{cfg.perturbation}.csv')
        csv_file2 = open(csv_dir2, 'w', newline='')
        writer2 = csv.writer(csv_file2)
        writer2.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                          'filename', "sen", "pre", "acc", "dice", "iou"])

        length = len(res_dicts)
        for i, res in enumerate(res_dicts):
            x_name, x, boxes, rects = res['x_name'], res['x'], res['pred_boxes'], res['pred_rects']

            index = pd.Index(score_file['filename']).get_loc(x_name)
            task = score_file['task'].get(index)

            img = np.moveaxis(re_normalize(x), source=0, destination=-1)

            img_dir = pathlib.Path(cfg.data_dir, cfg.dataset, task, f"seg_{cfg.seg_type}_whole_{cfg.perturbation}")
            img_dir.mkdir(parents=True, exist_ok=True)
            img_file_name = x_name.split(".")[0] + "_whole" + ".jpg"
            img_file_dir = pathlib.Path(img_dir, img_file_name)
            plt.imsave(img_file_dir, img)

            writer0.writerow(
                [score_file['refno'].get(index), score_file['visno'].get(index),
                 score_file['ethnic'].get(index),
                 int(score_file['cra'].get(index)),
                 int(score_file['dry'].get(index)),
                 int(score_file['ery'].get(index)),
                 int(score_file['exc'].get(index)),
                 int(score_file['exu'].get(index)),
                 int(score_file['lic'].get(index)),
                 int(score_file['oed'].get(index)),
                 x_name, img_file_name, img_file_dir, task,
                 ])

            print(f'{task.upper()}: {i}/{length} {img_file_name} created')

            crop_dir = pathlib.Path(cfg.data_dir, cfg.dataset, task, f"seg_{cfg.seg_type}_crops_{cfg.perturbation}")
            crop_dir.mkdir(parents=True, exist_ok=True)

            no = 0
            for id, box in enumerate(boxes):
                crops = crop_square(img, box, rects[id])
                for crop in crops:
                    crop_file_name = x_name.split(".")[0] + "_crop" + str(no) + ".jpg"
                    crop_file_dir = pathlib.Path(crop_dir, crop_file_name)
                    plt.imsave(crop_file_dir, crop)

                    writer1.writerow(
                        [score_file['refno'].get(index), score_file['visno'].get(index),
                         score_file['ethnic'].get(index),
                         int(score_file['cra'].get(index)),
                         int(score_file['dry'].get(index)),
                         int(score_file['ery'].get(index)),
                         int(score_file['exc'].get(index)),
                         int(score_file['exu'].get(index)),
                         int(score_file['lic'].get(index)),
                         int(score_file['oed'].get(index)),
                         x_name, crop_file_name, crop_file_dir, task,
                         ])

                    no += 1
                    print(f'{task.upper()}: {i}/{length} {crop_file_name} created')

            writer2.writerow(
                [score_file['refno'].get(index), score_file['visno'].get(index),
                 score_file['ethnic'].get(index),
                 int(score_file['cra'].get(index)),
                 int(score_file['dry'].get(index)),
                 int(score_file['ery'].get(index)),
                 int(score_file['exc'].get(index)),
                 int(score_file['exu'].get(index)),
                 int(score_file['lic'].get(index)),
                 int(score_file['oed'].get(index)),
                 x_name,
                 res["sen"], res["pre"], res["acc"], res["dice"], res["iou"],
                 ])

        csv_file1.close()
        csv_file2.close()

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
