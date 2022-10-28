import pyrootutils
import os
import pathlib

# root = pathlib.Path(os.path.abspath(''))
# pyrootutils.set_root(
#     path=root, # path to the root directory
#     project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
#     dotenv=True, # load environment variables from .env if exists in root directory
#     pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
#     cwd=True, # change current working directory to the root directory (helps with filepaths)
# )

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import hydra
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Tuple, Optional
from src.datamodules.components.transfroms import ComposeDouble, FunctionWrapperDouble, normalize_01, re_normalize
from src.datamodules.datasets.seg_dataset import SegDataSet
from src.utils.roi import draw_box_countours, save_json, crop_square
from src.utils.utils import get_classes, get_filenames_of_path
from src import utils

log = utils.get_pylogger(__name__)

def crop(cfg: DictConfig) -> Tuple[dict, dict]:
    SEGMENTATION = cfg.seg_type
    PERTURBATION = cfg.perturbation
    DATASET = cfg.dataset
    print("Program initiating... \nType of segmentation: " + SEGMENTATION + "\nType of perturbation: " + PERTURBATION)

    DATA_DIR = cfg.paths.data_dir
    CLASSES = get_classes(SEGMENTATION)
    SEG_TYPE = CLASSES.index(SEGMENTATION)-1

    score_dir = pathlib.Path(DATA_DIR, DATASET, 'score.csv')
    score_file = pd.read_csv(score_dir)

    csv_dir = pathlib.Path(DATA_DIR, DATASET, f'score_{SEGMENTATION}_crop_{PERTURBATION}.csv')
    csv_file = open(csv_dir, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['refno', 'visno', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed', 'filename', 'filepath'])

    for mode in ['train','valid','test']:

        # %% input and target files
        input_dir = pathlib.Path(DATA_DIR, DATASET, mode, "reals")
        target_dir = pathlib.Path(DATA_DIR, DATASET, mode, "labels")
        inputs = get_filenames_of_path(input_dir)
        targets = get_filenames_of_path(target_dir)
        inputs.sort()
        targets.sort()
        print('Reading images from: ' + str(input_dir))
        print('Reading masks from: ' + str(target_dir))

        # # debug
        # inputs = inputs[233:]
        # targets = targets[233:]

        box_dir = pathlib.Path(DATA_DIR, DATASET, mode, f"boxes_{SEGMENTATION}")
        box_dir.mkdir(parents=True, exist_ok=True)
        crop_dir = pathlib.Path(DATA_DIR, DATASET, mode, f"man_{SEGMENTATION}_crops_{PERTURBATION}")
        crop_dir.mkdir(parents=True, exist_ok=True)

        transforms = ComposeDouble([
            # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
            # AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
            # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        # %% dataset
        dataset = SegDataSet(inputs=inputs,
                             targets=targets,
                             transform=transforms,
                             seg_type=SEG_TYPE)

        # dataloader
        dataloader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False)

        for batch_id, [x, y, x_name, y_name] in enumerate(dataloader):
            x_name = x_name[0]

            # get boxes
            box_img, boxes, _ = draw_box_countours(y[0, SEG_TYPE].numpy(), rotated=False)
            box_img_rot, boxes_rot, rects_rot = draw_box_countours(y[0, SEG_TYPE].numpy(), rotated=True)

            # save boxes
            labels = [CLASSES[-1]]*len(boxes)
            json_file = {"labels": labels, "boxes": boxes}
            json_filename = x_name.split(".")[0] + ".json"
            save_json(json_file, path=pathlib.Path(box_dir, json_filename))

            print(f'{mode.upper()}: {batch_id}/{len(dataset)} {json_filename} created')

            try:
                index = pd.Index(score_file['filename']).get_loc(x_name)
            except:
                print(x_name + " not found")
                continue

            # save crops
            no = 0
            for i, box in enumerate(boxes_rot):
                img = np.moveaxis(re_normalize(x[0].numpy()), source=0, destination=-1)
                crops = crop_square(img, box, rects_rot[i])
                for crop in crops:
                    crop_file_name = x_name.split(".")[0] + "_crop" + str(no) + ".jpg"
                    crop_file_dir = pathlib.Path(crop_dir, crop_file_name)
                    plt.imsave(crop_file_dir, crop)

                    writer.writerow(
                        [score_file['refno'].get(index), score_file['visno'].get(index),
                         int(score_file['cra'].get(index)),
                         int(score_file['dry'].get(index)),
                         int(score_file['ery'].get(index)),
                         int(score_file['exc'].get(index)),
                         int(score_file['exu'].get(index)),
                         int(score_file['lic'].get(index)),
                         int(score_file['oed'].get(index)),
                         crop_file_name, crop_file_dir])

                    no += 1
                    print(f'{mode.upper()}: {batch_id}/{len(dataset)} {crop_file_name} created')

    csv_file.close()


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    crop(cfg)


if __name__ == "__main__":
    main()
