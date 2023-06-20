"""
To generate csv file of metadata

The keys blow (from original EczemaNet code) show how to choose signs for a specific score in SWET:
ALL_COMBINED_KEYS = {
        "sassad_tot": ['sassad_cra', 'sassad_dry', 'tiss_ery', 'tiss_exc', 'sassad_exu', 'sassad_lic'],
        "scorad_tot": ['tiss_ery', 'tiss_oed', 'sassad_exu', 'tiss_exc', 'sassad_lic', 'sassad_dry'],
        "easi_tot": ['tiss_ery', 'tiss_oed', 'tiss_exc', 'sassad_lic'],
        "tiss_tot": ['tiss_ery', 'tiss_oed', 'tiss_exc']
    }
"""

import pyrootutils

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

for DATASET in ['SWET', 'TLA']:

    ori_file_dir = pathlib.Path(DATA_DIR, DATASET, 'metadata_ori.csv')

    if DATASET == 'SWET':
        ori_file = pd.read_csv(ori_file_dir, index_col='refno')

        ethic_file_dir = pathlib.Path(DATA_DIR, DATASET, 'ethnic.csv')
        ethic_file = pd.read_csv(ethic_file_dir, index_col='refno')

        ori_file = ori_file.join(ethic_file, how="inner")
        ori_file = ori_file.reset_index(level=0)

        column = ori_file['filename'].copy()
        col = []
        for (index, colname) in enumerate(column):
            col.append(colname[:6] + colname[12:-11])
    elif DATASET == 'TLA':
        ori_file = pd.read_csv(ori_file_dir)

        column = ori_file['image'].copy()
        col = []
        for (index, colname) in enumerate(column):
            col.append(colname[11:-4])

    csv_dir = pathlib.Path(DATA_DIR, DATASET, 'metadata.csv')
    file = open(csv_dir, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['refno', 'visno', 'ethnic', 'cra', 'dry', 'ery', 'exc', 'exu', 'lic', 'oed',
                     'filename', 'filepath', 'task'])

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
                         ori_file['ethnic'].get(index).lower(),
                         # here to choose signs
                         int(ori_file['sassad_cra'].get(index)),
                         int(ori_file['sassad_dry'].get(index)),
                         int(ori_file['tiss_ery'].get(index)),
                         int(ori_file['tiss_exc'].get(index)),
                         int(ori_file['sassad_exu'].get(index)),
                         int(ori_file['sassad_lic'].get(index)),
                         int(ori_file['tiss_oed'].get(index)),
                         input.name, str(input), mode])
                elif DATASET == 'TLA':
                    if ori_file['status'].get(index) == 'Done':
                        writer.writerow(
                            [ori_file['id'].get(index), ori_file['visno'].get(index),
                             ori_file['ethnicity'].get(index).lower(),
                             int(ori_file['cra'].get(index)[-1]),
                             int(ori_file['dry'].get(index)[-1]),
                             int(ori_file['ery'].get(index)[-1]),
                             int(ori_file['exc'].get(index)[-1]),
                             int(ori_file['exu'].get(index)[-1]),
                             int(ori_file['lic'].get(index)[-1]),
                             int(ori_file['oed'].get(index)[-1]),
                             input.name, str(input), mode])
                    else:
                        print(input.name + " has no scores")
            except:
                print(input.name+" not found")

    file.close()