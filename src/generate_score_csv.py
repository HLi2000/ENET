import pyrootutils
import os
import pathlib

root = pathlib.Path(os.path.abspath(''))
pyrootutils.set_root(
    path=root, # path to the root directory
    project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
    dotenv=True, # load environment variables from .env if exists in root directory
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    cwd=True, # change current working directory to the root directory (helps with filepaths)
)

import pathlib
import pandas as pd
import csv

from src.utils.utils import get_filenames_of_path

DATA_DIR = root/"data"

for DATASET in ['SWET', 'TLA']:

    ori_file_dir = pathlib.Path(DATA_DIR, DATASET, 'metadata.csv')
    ori_file = pd.read_csv(ori_file_dir)

    if DATASET == 'SWET':
        column = ori_file['filename'].copy()
        col = []
        for (index, colname) in enumerate(column):
            col.append(colname[:6] + colname[12:-11])
    elif DATASET == 'TLA':
        column = ori_file['image'].copy()
        col = []
        for (index, colname) in enumerate(column):
            col.append(colname[11:-4])

    csv_dir = pathlib.Path(DATA_DIR, DATASET, 'score.csv')
    file = open(csv_dir, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['refno', 'visno', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed', 'filename', 'filepath'])

    for mode in ['train','valid','test']:
        input_dir = pathlib.Path(DATA_DIR, DATASET, mode, "reals")
        inputs = get_filenames_of_path(input_dir)
        inputs.sort()

        for input in inputs:
            try:
                index = col.index(input.name[:-4])
                if DATASET == 'SWET':
                    writer.writerow(
                        [ori_file['refno'].get(index), ori_file['visno'].get(index),
                         int(ori_file['sassad_cra'].get(index)),
                         int(ori_file['sassad_dry'].get(index)),
                         int(ori_file['tiss_ery'].get(index)),
                         int(ori_file['tiss_exc'].get(index)),
                         int(ori_file['sassad_exu'].get(index)),
                         int(ori_file['sassad_lic'].get(index)),
                         int(ori_file['tiss_oed'].get(index)),
                         input.name, str(input)])
                elif DATASET == 'TLA':
                    if ori_file['status'].get(index) == 'Done':
                        writer.writerow(
                            [ori_file['id'].get(index), ori_file['visno'].get(index),
                             int(ori_file['cra'].get(index)[-1]),
                             int(ori_file['dry'].get(index)[-1]),
                             int(ori_file['ery'].get(index)[-1]),
                             int(ori_file['exc'].get(index)[-1]),
                             int(ori_file['exu'].get(index)[-1]),
                             int(ori_file['lic'].get(index)[-1]),
                             int(ori_file['oed'].get(index)[-1]),
                             input.name, str(input)])
                    else:
                        print(input.name + " has no scores")
            except:
                print(input.name+" not found")

    file.close()






