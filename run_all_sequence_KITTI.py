import glob
import os
import pandas as pd
import numpy as np
from track_main import main_cyclic
from configuration import get_config


# TRACKER_NAME = 'customSORT_priorityAssociation'
TRACKER_NAME = 'customSORT_pa_lowr_exfov'
DET_FOLDER = f"data/nms/"
DATASET = 'KITTI'
SAVE_PATH = f'data/tracks/{TRACKER_NAME}/data/'


if __name__ == "__main__":
    config = get_config(DATASET)
    det_list = [f for f in glob.glob(f"{DET_FOLDER}*.csv")]
    fps = config['FPS']
    os.makedirs(SAVE_PATH, exist_ok = True)
    cycle_time = 1/fps

    for k, det_seq in enumerate(det_list):
        print(f"{k+1}/{len(det_list)}")
        seq_name = det_seq[-8:-4]

        tracks_df = main_cyclic(det_seq, DATASET)
        
        tracks_df.to_csv(f"{SAVE_PATH}{seq_name}.txt", index=None, header=None, sep=' ')