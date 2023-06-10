import random

import torch
import pathlib

from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from skimage.io import imread
from typing import List, Dict
from src.datamodules.components.transfroms import ComposeDouble


class PredDataSet(torch.utils.data.Dataset):
    """
    Builds a dataset with images and their respective targets.
    A target is expected to be a pickled file of a dict
    and should contain at least a 'boxes' and a 'labels' key.
    inputs and targets are expected to be a list of pathlib.Path objects.
    In case your labels are strings, you can use mapping (a dict) to int-encode them.
    Returns a dict with the following keys: 'x', 'x_name', 'y', 'y_name'
    """

    def __init__(self,
                 inputs: List[str],
                 targets: List[Dict],
                 transform: ComposeDouble = None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):

        # Load input and target
        x_path = self.inputs[index]
        x = imread(x_path)
        target = self.targets[index]

        # From RGBA to RGB
        if x.shape[-1] == 4:
            from skimage.color import rgba2rgb
            x = rgba2rgb(x)

        if self.transform is not None:
            x, target = self.transform(x, target)  # returns np.ndarray

        # plt.imsave(f'x_{random.randint(0, 10000)}.jpg', x.T)

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        target = {key: torch.tensor(value).type(torch.float32) for key, value in target.items()}

        return {'x': x, 'y': target, 'x_name': x_path, 'y_name': x_path}