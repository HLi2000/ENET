import pathlib
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
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms_train = ComposeDouble([
            PredAlbumentationWrapper(albumentations=[
                A.Resize(256, 256, always_apply=True),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomScale(p=0.5, scale_limit=0.5)
            ]),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        self.transforms_valid = ComposeDouble([
            PredAlbumentationWrapper(albumentations=[
                A.Resize(256, 256, always_apply=True),
            ]),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        self.transforms_test = ComposeDouble([
            PredAlbumentationWrapper(albumentations=[
                A.Resize(256, 256, always_apply=True),
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
            auto_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_auto_crop_{self.hparams.perturbation}.csv"
            auto_csv = pd.read_csv(auto_csv_dir)
            man_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_man_crop_{self.hparams.perturbation}.csv"
            man_csv = pd.read_csv(man_csv_dir)

            csv = pd.concat([auto_csv, man_csv])
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
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.hparams.crop:
                auto_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_auto_crop_{self.hparams.perturbation}.csv"
                auto_csv = pd.read_csv(auto_csv_dir)
                man_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_man_crop_{self.hparams.perturbation}.csv"
                man_csv = pd.read_csv(man_csv_dir)

                # csv = pd.concat([auto_csv, man_csv])
                csv = man_csv
                csv_train = csv[csv['task'] == 'train'][:10]
                csv_val = csv[csv['task'] == 'valid'][:10]

                inputs_train = [filepath for filepath in csv_train['filepath']]
                targets_train = csv_train[['cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed']].to_dict('records')

                inputs_val = [filepath for filepath in csv_val['filepath']]
                targets_val = csv_val[['cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed']].to_dict('records')

                if inputs_train and inputs_val:
                    log.info(f'{len(auto_csv)} crops loaded from {auto_csv_dir}')
                    log.info(f'{len(man_csv)} crops loaded from {man_csv_dir}')
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
            if self.hparams.crop:
                # auto_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_pred_{self.hparams.perturbation}.csv"
                auto_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_man_crop_{self.hparams.perturbation}.csv"
                auto_csv = pd.read_csv(auto_csv_dir)
                # man_csv_dir = self.data_dir / f"metadata_{self.hparams.seg_type}_man_crop_{self.hparams.perturbation}.csv"
                # man_csv = pd.read_csv(man_csv_dir)

                # csv = pd.concat([auto_csv, man_csv])
                csv = auto_csv
                # csv_test = csv[csv['task'] == 'test']
                csv_test = csv[csv['task'] == 'train'][:10]

                inputs_test = [filepath for filepath in csv_test['filepath']]
                targets_test = csv_test[['cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed']].to_dict('records')

                if inputs_test:
                    log.info(f'{len(auto_csv)} crops loaded from {auto_csv_dir}')
                    # log.info(f'{len(man_csv)} crops loaded from {man_csv_dir}')
                    log.info(f'Test: {len(inputs_test)} crops')
                else:
                    raise ValueError('Wrong csv files!')

                self.dataset_test = PredDataSet(inputs=inputs_test,
                                               targets=targets_test,
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
