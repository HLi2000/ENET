import cv2
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class SegDataSet(torch.utils.data.Dataset):
    """
    Builds a dataset with images and their respective targets.
    """

    def __init__(self,
                 inputs: List[pathlib.Path],
                 targets: List[pathlib.Path],
                 transform = None,
                 seg_type = 0
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.seg_type = seg_type

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_path = self.inputs[index]
        target_path = self.targets[index]

        # Load input and target
        x, y = self.read_images(input_path, target_path)

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        # plt.imshow(x)
        # plt.show()

        y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
        skin = y > 64
        ad = y > 192

        if self.seg_type == 0:
            y = np.expand_dims(skin, axis=-1).astype(np.uint8)
        else:
            y = np.expand_dims(ad, axis=-1).astype(np.uint8)

        if self.transform is not None:
            x, y = self.transform(x, y)  # returns np.ndarrays

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        y = torch.from_numpy(y).type(torch.float32)

        return {'x': x, 'y': y, 'x_name': self.inputs[index].name}

    @staticmethod
    def read_images(inp, tar):
        return cv2.imread(str(inp)), cv2.imread(str(tar))
