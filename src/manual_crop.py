"""
To generate manual crops/boxes/segmentations for training ROI detection networks and inputting to prediction network
"""

import pyrootutils
import pathlib

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
from src.utils.seg import collate_double_seg

log = utils.get_pylogger(__name__)

def crop(cfg: DictConfig) -> Tuple[dict, dict]:
    SEGMENTATION = 'skin' # 'skin' or 'ad'
    PERTURBATION = 'base'
    DATASET = 'TLA'
    METHOD = 'whole' # 'whole' or 'seg' or 'ROI'
    # 'whole' for whole segmentation, 'seg' for background-removed crops, 'ROI' for crops with backgrounds
    print("Program initiating... \nType of segmentation: " + SEGMENTATION + "\nType of perturbation: " + METHOD)

    DATA_DIR = cfg.paths.data_dir
    CLASSES = get_classes(SEGMENTATION)
    SEG_TYPE = CLASSES.index(SEGMENTATION)-1

    score_dir = pathlib.Path(DATA_DIR, DATASET, 'metadata.csv')
    score_file = pd.read_csv(score_dir)

    if METHOD == 'whole':
        csv_dir = pathlib.Path(DATA_DIR, DATASET, f'metadata_{SEGMENTATION}_man_whole_seg.csv')
    else:
        csv_dir = pathlib.Path(DATA_DIR, DATASET, f'metadata_{SEGMENTATION}_man_crop_{METHOD}.csv')
    csv_file = open(csv_dir, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                     'filename', 'crop', 'filepath', 'task'])

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
        # inputs = inputs[537:]
        # targets = targets[537:]

        if METHOD == 'whole':
            img_dir = pathlib.Path(DATA_DIR, DATASET, mode, f"man_{SEGMENTATION}_whole_seg")
            img_dir.mkdir(parents=True, exist_ok=True)
        else:
            box_dir = pathlib.Path(DATA_DIR, DATASET, mode, f"boxes_{SEGMENTATION}")
            box_dir.mkdir(parents=True, exist_ok=True)
            crop_dir = pathlib.Path(DATA_DIR, DATASET, mode, f"man_{SEGMENTATION}_crops_{METHOD}")
            crop_dir.mkdir(parents=True, exist_ok=True)

        transforms = ComposeDouble([
            # ROIAlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
            # ROIAlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
            # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0, target=True),
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
                                shuffle=False,
                                collate_fn=collate_double_seg)

        for batch_id, [x, y, x_name] in enumerate(dataloader):

            if METHOD == 'whole':
                x = x * y
                img = np.moveaxis(re_normalize(x[0].numpy()), source=0, destination=-1)

                x_name = x_name[0]

                try:
                    index = pd.Index(score_file['filename']).get_loc(x_name)
                except:
                    print(x_name + " not found")
                    continue

                file_name = x_name.split(".")[0] + "_whole" + ".jpg"
                file_dir = pathlib.Path(img_dir, file_name)
                plt.imsave(file_dir, img)

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
                     x_name, file_name, file_dir, mode,
                     ])

                print(f'{mode.upper()}: {batch_id}/{len(dataset)} {file_name} created')

                continue

            if METHOD == 'seg':
                x = x * y

            x_name = x_name[0]

            # get boxes
            box_img, boxes, rects = draw_box_countours(y[0, 0].numpy(), rotated=False)
            box_img_rot, boxes_rot, rects_rot = draw_box_countours(y[0, 0].numpy(), rotated=True)

            # save boxes
            if len(boxes) == 0:
                print(f'{mode.upper()}: {batch_id}/{len(dataset)} {json_filename} is empty!!!!!')
                boxes.append([0,0,1,1])
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

            img = np.moveaxis(re_normalize(x[0].numpy()), source=0, destination=-1)
            # save crops
            no = 0
            if METHOD == 'seg':
                for i, box in enumerate(boxes_rot):
                    crops = crop_square(img, box, rects_rot[i])
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
                             x_name, crop_file_name, crop_file_dir, mode,
                             ])

                        no += 1
                        print(f'{mode.upper()}: {batch_id}/{len(dataset)} {crop_file_name} created')
            else:
                for i, box in enumerate(boxes):
                    x, y, x2, y2 = box
                    crop = img[int(y):int(y2), int(x):int(x2)]
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
                         x_name, crop_file_name, crop_file_dir, mode,
                         ])

                    no += 1
                    print(f'{mode.upper()}: {batch_id}/{len(dataset)} {crop_file_name} created')

    csv_file.close()


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    crop(cfg)


if __name__ == "__main__":
    main()
