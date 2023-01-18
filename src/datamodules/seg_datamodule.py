import pathlib
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import albumentations as A
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.transfroms import ComposeDouble, SegAlbumentationWrapper, FunctionWrapperDouble, \
    normalize_01
from src.datamodules.datasets.seg_dataset import SegDataSet
from src.utils.seg import collate_double_seg
from src.utils.utils import get_filenames_of_path, get_classes

from src import utils

log = utils.get_pylogger(__name__)

class SegDataModule(LightningDataModule):
    """LightningDataModule for ROI.

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
        seg_type: str = "skin",
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        auto_crop: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        classes = get_classes(seg_type)
        self.seg_id = classes.index(seg_type) - 1

        # data transformations
        # %% transformations and augmentations
        self.transforms_train = ComposeDouble([
            SegAlbumentationWrapper(albumentations=[
                A.Resize(1024, 768, always_apply=True),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomScale(p=0.5, scale_limit=0.5)
            ]),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0, target=True),
            FunctionWrapperDouble(normalize_01)
        ])

        self.transforms_val = ComposeDouble([
            SegAlbumentationWrapper(albumentations=[
                A.Resize(1024, 768, always_apply=True),
            ]),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0, target=True),
            FunctionWrapperDouble(normalize_01)
        ])

        self.transforms_test = ComposeDouble([
            SegAlbumentationWrapper(albumentations=[
                A.Resize(1024, 768, always_apply=True),
            ]),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0, target=True),
            FunctionWrapperDouble(normalize_01)
        ])

        self.data_dir = pathlib.Path(data_dir)
        self.dataset_train: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 2

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
            inputs_train_dir = self.data_dir / 'train' / "reals"
            targets_train_dir = self.data_dir / 'train' / "labels"
            inputs_train = get_filenames_of_path(inputs_train_dir)
            targets_train = get_filenames_of_path(targets_train_dir)
            inputs_train.sort()
            targets_train.sort()
            if inputs_train:
                log.info(f'{len(inputs_train)} images loaded from {inputs_train_dir}')
                log.info(f'{len(targets_train)} masks loaded from {targets_train_dir}')
            else:
                raise ValueError('Wrong path!')

            inputs_val_dir = self.data_dir / 'valid' / "reals"
            targets_val_dir = self.data_dir / 'valid' / "labels"
            inputs_val = get_filenames_of_path(inputs_val_dir)
            targets_val = get_filenames_of_path(targets_val_dir)
            inputs_val.sort()
            targets_val.sort()
            if inputs_val:
                log.info(f'{len(inputs_val)} images loaded from {inputs_val_dir}')
                log.info(f'{len(targets_val)} masks loaded from {targets_val_dir}')
            else:
                raise ValueError('Wrong path!')


            self.dataset_train = SegDataSet(inputs=inputs_train,
                                            targets=targets_train,
                                            transform=self.transforms_train,
                                            seg_type=self.seg_id)

            self.dataset_val = SegDataSet(inputs=inputs_val,
                                          targets=targets_val,
                                          transform=self.transforms_val,
                                          seg_type=self.seg_id)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            if self.hparams.auto_crop:
                inputs_train_dir = self.data_dir / 'train' / "reals"
                targets_train_dir = self.data_dir / 'train' / "labels"
                inputs_train = get_filenames_of_path(inputs_train_dir)
                targets_train = get_filenames_of_path(targets_train_dir)
                inputs_train.sort()
                targets_train.sort()

                inputs_val_dir = self.data_dir / 'valid' / "reals"
                targets_val_dir = self.data_dir / 'valid' / "labels"
                inputs_val = get_filenames_of_path(inputs_val_dir)
                targets_val = get_filenames_of_path(targets_val_dir)
                inputs_val.sort()
                targets_val.sort()

                inputs_test = inputs_train + inputs_val
                targets_test = targets_train + targets_val
                inputs_test_dir = str(inputs_train_dir)+' and '+str(inputs_train_dir)
                targets_test_dir = str(targets_val_dir)+' and '+str(targets_val_dir)
            else:
                inputs_test_dir = self.data_dir / 'test' / "reals"
                targets_test_dir = self.data_dir / 'test' / "labels"
                inputs_test = get_filenames_of_path(inputs_test_dir)
                targets_test = get_filenames_of_path(targets_test_dir)
                inputs_test.sort()
                targets_test.sort()
            if inputs_test:
                log.info(f'{len(inputs_test)} images loaded from {inputs_test_dir}')
                log.info(f'{len(targets_test)} masks loaded from {targets_test_dir}')
            else:
                raise ValueError('Wrong path!')

            self.dataset_test = SegDataSet(inputs=inputs_test,
                                           targets=targets_test,
                                           transform=self.transforms_test,
                                           seg_type=self.seg_id)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_double_seg,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_double_seg,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_double_seg,
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "roi.yaml")
    _ = hydra.utils.instantiate(cfg)
