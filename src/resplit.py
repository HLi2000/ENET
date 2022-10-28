import shutil

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

import pathlib
import pandas as pd
import csv

from src.utils.utils import get_filenames_of_path

DATA_DIR = root/"data"

for DATASET in ['SWET_old', 'TLA_old']:

    ori_file_dir = pathlib.Path(DATA_DIR, DATASET, 'metadata_old_revised.csv')
    ori_file = pd.read_csv(ori_file_dir)

    column = ori_file['filename'].copy()
    col = []
    for (index, colname) in enumerate(column):
        col.append(colname)

    for mode in ['train','valid','test']:
        input_dir = pathlib.Path(DATA_DIR, DATASET, mode, "reals")
        inputs = get_filenames_of_path(input_dir)
        inputs.sort()
        target_dir = pathlib.Path(DATA_DIR, DATASET, mode, "labels")
        targets = get_filenames_of_path(target_dir)
        targets.sort()

        length = len(inputs)
        for i, input in enumerate(inputs):
            try:
                index = col.index(input.name)
                task = ori_file['task'].get(index)
                input_dst = pathlib.Path(DATA_DIR, DATASET[:-4], task, "reals")
                input_dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(input), str(input_dst))
                target_dst = pathlib.Path(DATA_DIR, DATASET[:-4], task, "labels")
                target_dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(targets[i]), str(target_dst))
                print(f"{DATASET} {mode} {i+1}/{length}")
            except ValueError:
                print(input.name+" not found")
