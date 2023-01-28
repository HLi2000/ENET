import csv
import pathlib
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.datamodules.components.transfroms import re_normalize
from src.utils.roi import crop_square, crop_rect
from src.utils.utils import TestStore

from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

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

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

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

    metric_dict = trainer.callback_metrics
    res_dicts = test_store.get_test_outputs()

    score_dir = pathlib.Path(cfg.data_dir, cfg.dataset, 'metadata.csv')
    score_file = pd.read_csv(score_dir)

    if cfg.auto_crop:
        csv_dir = pathlib.Path(cfg.data_dir, cfg.dataset, f'metadata_{cfg.seg_type}_ROI_crop_{cfg.perturbation}.csv')
        csv_file = open(csv_dir, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                         'filename', 'crop', 'filepath', 'task'])

        length = len(res_dicts)
        print(str(length)+'!!!!!!!!!!!!!!!!!!!')
        for i, res in enumerate(res_dicts):
            x_name, x, boxes = res['x_name'], res['x'], res['pred_boxes']

            index = pd.Index(score_file['filename']).get_loc(x_name)
            task = score_file['task'].get(index)

            img = np.moveaxis(re_normalize(x), source=0, destination=-1)

            crop_dir = pathlib.Path(cfg.data_dir, cfg.dataset, task, f"ROI_{cfg.seg_type}_crops_{cfg.perturbation}")
            crop_dir.mkdir(parents=True, exist_ok=True)

            no = 0
            for id, box in enumerate(boxes['boxes']):
                if boxes['scores'][id] < 0.75:
                    continue
                crops = crop_square(img, box.cpu().numpy())
                for crop in crops:
                    crop_file_name = x_name.split(".")[0] + "_crop" + str(no) + ".jpg"
                    crop_file_dir = pathlib.Path(crop_dir, crop_file_name)
                    try:
                        plt.imsave(crop_file_dir, crop)
                    except:
                        continue

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

            if no == 0:
                for id, box in enumerate(boxes['boxes']):
                    if boxes['scores'][id] < 0.5:
                        continue
                    # crops = crop_square(img, box.cpu().numpy())
                    crops = [crop_rect(img, box.cpu().numpy())]
                    for crop in crops:
                        crop_file_name = x_name.split(".")[0] + "_crop" + str(no) + ".jpg"
                        crop_file_dir = pathlib.Path(crop_dir, crop_file_name)
                        try:
                            plt.imsave(crop_file_dir, crop)
                        except:
                            continue

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
            if no == 0:
                crop_file_name = x_name.split(".")[0] + "_crop" + str(no) + ".jpg"
                crop_file_dir = pathlib.Path(crop_dir, crop_file_name)
                plt.imsave(crop_file_dir, img)

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
        csv_dir1 = pathlib.Path(cfg.data_dir, cfg.dataset, f'metadata_{cfg.seg_type}_ROI_pred_{cfg.perturbation}.csv')
        csv_file1 = open(csv_dir1, 'w', newline='')
        writer1 = csv.writer(csv_file1)
        writer1.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                         'filename', 'crop', 'filepath', 'task'])

        csv_dir2 = pathlib.Path(cfg.data_dir, cfg.dataset, f'results_{cfg.seg_type}_ROI_{cfg.perturbation}.csv')
        csv_file2 = open(csv_dir2, 'w', newline='')
        writer2 = csv.writer(csv_file2)
        writer2.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                         'filename', "mAP", "mAP_50", "mAP_75", "sen", "pre", "acc", "dice", "iou"])

        length = len(res_dicts)
        for i, res in enumerate(res_dicts):
            x_name, x, boxes = res['x_name'], res['x'], res['pred_boxes']

            index = pd.Index(score_file['filename']).get_loc(x_name)
            task = score_file['task'].get(index)

            img = np.moveaxis(re_normalize(x), source=0, destination=-1)

            crop_dir = pathlib.Path(cfg.data_dir, cfg.dataset, task, f"ROI_{cfg.seg_type}_crops_{cfg.perturbation}")
            crop_dir.mkdir(parents=True, exist_ok=True)

            no = 0
            for id, box in enumerate(boxes['boxes']):
                if boxes['scores'][id] < 0.75:
                    continue
                # crops = crop_square(img, box.cpu().numpy())
                crops = [crop_rect(img, box.cpu().numpy())]
                for crop in crops:
                    crop_file_name = x_name.split(".")[0] + "_crop" + str(no) + ".jpg"
                    crop_file_dir = pathlib.Path(crop_dir, crop_file_name)
                    try:
                        plt.imsave(crop_file_dir, crop)
                    except:
                        continue

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

            if no == 0:
                for id, box in enumerate(boxes['boxes']):
                    if boxes['scores'][id] < 0.5:
                        continue
                    # crops = crop_square(img, box.cpu().numpy())
                    crops = [crop_rect(img, box.cpu().numpy())]
                    for crop in crops:
                        crop_file_name = x_name.split(".")[0] + "_crop" + str(no) + ".jpg"
                        crop_file_dir = pathlib.Path(crop_dir, crop_file_name)
                        try:
                            plt.imsave(crop_file_dir, crop)
                        except:
                            continue

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
            if no == 0:
                crop_file_name = x_name.split(".")[0] + "_crop" + str(no) + ".jpg"
                crop_file_dir = pathlib.Path(crop_dir, crop_file_name)
                plt.imsave(crop_file_dir, img)

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
                 x_name, res['mAP'], res['mAP_50'], res['mAP_75'],
                 res["sen"], res["pre"], res["acc"], res["dice"], res["iou"],
                 ])

        csv_file1.close()
        csv_file2.close()

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
