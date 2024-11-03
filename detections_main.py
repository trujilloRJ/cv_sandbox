import cv2 as cv
import pandas as pd
import numpy as np
from configuration import get_config
from association import compute_iou

SEQUENCE = "0000"
DET_FILE = f"C:/javier/personal_projects/computer_vision/data/KITTI_object_tracking/detections_regionlet/training/det_02/{SEQUENCE}.txt"
DATASET = 'KITTI'

if __name__ == "__main__":
    config = get_config(DATASET)
    min_score = -1.0
    nms_iou_thr = 0.4

    # load dets
    config = get_config(DATASET)
    dets = pd.read_csv(DET_FILE, header=None, sep=" ")
    dets.columns = config['det_cols']

    # filter dets with low score
    dets = dets[dets['score'] > min_score]

    # sort by score
    dets = dets.sort_values(by='score', ascending=False).reset_index(drop=True)

    # perform NMS 
    picked_dets_indexes = []
    frames = dets['frame'].unique()
    for i, frame in enumerate(frames):
        suppress_index = []
        cur_dets = dets.loc[dets['frame'] == frame, :]
        cur_indexes = list(cur_dets.index)
        for j, j_idx in enumerate(cur_indexes):   # already sorted by score

            if j_idx in suppress_index: continue

            picked_dets_indexes.append(j_idx)
            box_ref = np.array([cur_dets.loc[j_idx, 'top'], cur_dets.loc[j_idx, 'left'], cur_dets.loc[j_idx, 'bottom'], cur_dets.loc[j_idx, 'right']])

            for k_idx in cur_indexes[j+1:]:

                if k_idx in suppress_index: continue

                box_test = np.array([cur_dets.loc[k_idx, 'top'], cur_dets.loc[k_idx, 'left'], cur_dets.loc[k_idx, 'bottom'], cur_dets.loc[k_idx, 'right']])
                iou = compute_iou(box_ref, box_test)
                if iou > nms_iou_thr:
                    suppress_index.append(k_idx)
    nms_dets = dets.loc[picked_dets_indexes, :]

    nms_dets.to_csv(f'{SEQUENCE}_nms_dets.csv', sep=" ", index=False, header=False)