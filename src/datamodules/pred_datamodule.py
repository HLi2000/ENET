import pathlib
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import albumentations as A
from pytorch_lightning import LightningDataModule
from sklearn.utils import compute_class_weight
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.datamodules.components.transfroms import ComposeDouble, FunctionWrapperDouble, normalize_01, \
    PredAlbumentationWrapper

from src import utils
from src.datamodules.datasets.pred_dataset import PredDataSet
from src.utils.roi import collate_double

log = utils.get_pylogger(__name__)

class PredDataModule(LightningDataModule):
    """Example of LightningDataModule for Prediction.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        crop: bool = False,
        seg_type: str = "skin",
        perturbation: str = "base",
        ordinal: bool = True,
        class_weighted: bool = False,
        method: str = 'ROI',
        num_folds: int = 5,
        k: int = 0,
    ):
        super().__init__()

        self.k = k
        self.num_folds = num_folds

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms_train = ComposeDouble([
            PredAlbumentationWrapper(albumentations=[
                # A.Resize(256, 256, always_apply=True),
                # A.LongestMaxSize(max_size=256),
                # A.PadIfNeeded(min_height=256, min_width=256, border_mode=4, value=(0, 0, 0)),
                A.LongestMaxSize(max_size=320),
                A.PadIfNeeded(min_height=320, min_width=320, border_mode=0, value=(0, 0, 0)),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(scale=(0.75, 1.25), p=0.5),
                A.Affine(rotate=(-90, 90), p=0.5),

                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), p=0.5),
                A.GaussianBlur(blur_limit=5, p=0.5),
                A.MotionBlur(blur_limit=5, p=0.5),
                A.RandomBrightness(limit=(-0.25, 0.25), p=0.5),
                A.RandomContrast(limit=(-0.25, 0.25), p=0.5),
                A.Downscale(scale_min=0.75, scale_max=0.95, interpolation=0, p=0.5)
            ]),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        self.transforms_valid = ComposeDouble([
            PredAlbumentationWrapper(albumentations=[
                # A.Resize(256, 256, always_apply=True),
                # A.LongestMaxSize(max_size=256),
                # A.PadIfNeeded(min_height=256, min_width=256, border_mode=4, value=(0, 0, 0)),
                A.LongestMaxSize(max_size=320),
                A.PadIfNeeded(min_height=320, min_width=320, border_mode=0, value=(0, 0, 0)),
            ]),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        self.transforms_test = ComposeDouble([
            PredAlbumentationWrapper(albumentations=[
                # A.Resize(256, 256, always_apply=True),
                # A.LongestMaxSize(max_size=256),
                # A.PadIfNeeded(min_height=256, min_width=256, border_mode=4, value=(0, 0, 0)),
                A.LongestMaxSize(max_size=320),
                A.PadIfNeeded(min_height=320, min_width=320, border_mode=0, value=(0, 0, 0)),

                # A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                # A.ISONoise(color_shift=(0.01, 0.05), p=1),
                # A.GaussianBlur(blur_limit=5, p=1),
                # A.MotionBlur(blur_limit=5, p=1),
                # A.RandomBrightness(limit=(-0.75, 0.75), p=1),
                # A.RandomContrast(limit=(-0.75, 0.75), p=1),
                # A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=0, p=1),
            ]),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        self.data_dir = pathlib.Path(data_dir)
        self.dataset_train: Optional[Dataset] = None
        self.dataset_valid: Optional[Dataset] = None
        self.dataset_test: Optional[Dataset] = None

    @property
    def class_weights(self):
        if self.hparams.class_weighted:
            auto_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_{self.hparams.method}_crop_{self.hparams.perturbation}.csv"
            auto_csv = pd.read_csv(auto_csv_dir)
            man_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_man_crop_{self.hparams.perturbation}.csv"
            man_csv = pd.read_csv(man_csv_dir)

            csv = pd.concat([auto_csv, man_csv])
            # csv = man_csv
            csv_train = csv[csv['task'] == 'train']

            weights_test = []
            for sign in ['cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed']:
                counts = csv_train.groupby(sign)['filepath'].nunique()
                weights = []
                for score in range(4):
                    count = counts.get(score)
                    if count:
                        weights.append(count)
                    else:
                        weights.append(1)
                if self.hparams.ordinal:
                    weights = [sum(weights[1:]), sum(weights[2:]), weights[3]]
                weights = min(weights) / np.array(weights)
                weights_test.append(weights)
            # weights_test = min(weights_test) / np.array(weights_test)
            return torch.tensor(weights_test).type(torch.float32)
        return None

    @property
    def num_classes(self):
        return 7

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if self.hparams.crop:
            auto_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_{self.hparams.method}_crop_{self.hparams.perturbation}.csv"
            auto_csv = pd.read_csv(auto_csv_dir)

            pred_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_{self.hparams.method}_pred_{self.hparams.perturbation}.csv"
            pred_csv = pd.read_csv(pred_csv_dir)

            auto_csv = pd.concat([auto_csv, pred_csv])

            auto_patients = [x for _, x in auto_csv.groupby('refno')]
            auto_folds = split_list(auto_patients, self.num_folds)
            # print(auto_folds)

            auto_csv_val = pd.concat(auto_folds[self.k])
            auto_csv_train = pd.concat(sum([x for i, x in enumerate(auto_folds) if i != self.k], []))

            man_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_man_crop_{self.hparams.method}.csv"
            man_csv = pd.read_csv(man_csv_dir)

            man_patients = [x for _, x in man_csv.groupby('refno')]
            man_folds = split_list(man_patients, self.num_folds)
            # print(auto_folds)

            # man_csv_val = pd.concat(man_folds[self.k])
            man_csv_train = pd.concat(sum([x for i, x in enumerate(man_folds) if i != self.k], []))

            csv_val = auto_csv_val
            csv_train = pd.concat([auto_csv_train, man_csv_train])
            # csv = man_csv
        else:
            if self.hparams.method == 'seg':
                auto_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_{self.hparams.method}_whole_{self.hparams.perturbation}.csv"
                auto_csv = pd.read_csv(auto_csv_dir)

                pred_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_{self.hparams.method}_pred_whole_{self.hparams.perturbation}.csv"
                pred_csv = pd.read_csv(pred_csv_dir)

                auto_csv = pd.concat([auto_csv, pred_csv])

                auto_patients = [x for _, x in auto_csv.groupby('refno')]
                auto_folds = split_list(auto_patients, self.num_folds)

                auto_csv_val = pd.concat(auto_folds[self.k])
                auto_csv_train = pd.concat(sum([x for i, x in enumerate(auto_folds) if i != self.k], []))

                man_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_man_whole_seg.csv"
                man_csv = pd.read_csv(man_csv_dir)

                man_patients = [x for _, x in man_csv.groupby('refno')]
                man_folds = split_list(man_patients, self.num_folds)

                # man_csv_val = pd.concat(man_folds[self.k])
                man_csv_train = pd.concat(sum([x for i, x in enumerate(man_folds) if i != self.k], []))

                csv_val = auto_csv_val
                csv_train = pd.concat([auto_csv_train, man_csv_train])
            else:
                csv_dir = self.data_dir / f"metadata.csv"
                csv = pd.read_csv(csv_dir)

                patients = [x for _, x in csv.groupby('refno')]
                folds = split_list(patients, self.num_folds)

                csv_val = pd.concat(folds[self.k])
                csv_train = pd.concat(sum([x for i, x in enumerate(folds) if i != self.k], []))
                # csv_val = csv_train

        inputs_train = [filepath for filepath in csv_train['filepath']]
        targets_train = csv_train[['cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed']].to_dict('records')

        inputs_val = [filepath for filepath in csv_val['filepath']]
        targets_val = csv_val[['cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed']].to_dict('records')

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:

            if inputs_train and inputs_val:
                if self.hparams.crop:
                    log.info(f'{len(auto_csv)} crops loaded from {auto_csv_dir} and {pred_csv_dir}')
                    log.info(f'{len(man_csv)} crops loaded from {man_csv_dir}')
                else:
                    if self.hparams.method == 'seg':
                        log.info(f'{len(auto_csv)} crops loaded from {auto_csv_dir} and {pred_csv_dir}')
                        log.info(f'{len(man_csv)} crops loaded from {man_csv_dir}')
                    else:
                        log.info(f'{len(csv)} crops loaded from {csv_dir}')
                log.info(f'Train: {len(inputs_train)} crops')
                log.info(f'Valid: {len(inputs_val)} crops')
            else:
                raise ValueError('Wrong csv files!')

            self.dataset_train = PredDataSet(inputs=inputs_train,
                                            targets=targets_train,
                                            transform=self.transforms_train)

            self.dataset_val = PredDataSet(inputs=inputs_val,
                                          targets=targets_val,
                                          transform=self.transforms_valid)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:

            inputs_val = [filepath for filepath in csv_val['filepath']]
            targets_val = csv_val[['cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed']].to_dict('records')

            if inputs_val:
                if self.hparams.crop:
                    log.info(f'{len(auto_csv)} crops loaded from {auto_csv_dir} and {pred_csv_dir}')
                    # log.info(f'{len(man_csv)} crops loaded from {man_csv_dir}')
                else:
                    if self.hparams.method == 'seg':
                        log.info(f'{len(auto_csv)} crops loaded from {auto_csv_dir} and {pred_csv_dir}')
                        log.info(f'{len(man_csv)} crops loaded from {man_csv_dir}')
                    else:
                        log.info(f'{len(csv)} crops loaded from {csv_dir}')
                log.info(f'Test: {len(inputs_val)} crops')
            else:
                raise ValueError('Wrong csv files!')

            self.dataset_test = PredDataSet(inputs=inputs_val,
                                           targets=targets_val,
                                           transform=self.transforms_test)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_double,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_double,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_double,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "pred.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)

def split_list(a, n):
    random.seed(37)
    random.shuffle(a)
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]