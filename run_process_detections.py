import glob
import cv2 as cv
import pandas as pd
import numpy as np
from configuration import get_config
from association import compute_iou
from utils import OBJ_TYPES

DET_FOLDER = f"data/detections_regionlet/training/det_02/"
DATASET = 'KITTI'
SAVE_PATH = f"data/nms/"

if __name__ == "__main__":
    config = get_config(DATASET)
    det_list = [f for f in glob.glob(f"{DET_FOLDER}*.txt")]
    min_score = -1.0
    nms_iou_thr = 0.4
    interested_types = OBJ_TYPES


    for k, det_seq in enumerate(det_list):
        print(f"{k+1}/{len(det_list)}")

        # load dets
        dets = pd.read_csv(det_seq, header=None, sep=" ")
        dets.columns = config['det_cols']

        # filter dets with low score and type
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

        # post-process detections
        nms_dets.loc[:, 'center_x'] = (nms_dets.loc[:, 'top'] + nms_dets.loc[:, 'bottom']) // 2
        nms_dets.loc[:, 'center_y'] = (nms_dets.loc[:, 'left'] + nms_dets.loc[:, 'right']) // 2
        nms_dets.loc[:, 'width'] = (nms_dets.loc[:, 'right'] - nms_dets.loc[:, 'left']).astype(int)
        nms_dets.loc[:, 'height'] = (nms_dets.loc[:, 'bottom'] - nms_dets.loc[:, 'top']).astype(int)

        seq_name = det_seq[-8:-4]
        nms_dets = nms_dets.sort_values(by='frame', ascending=True).reset_index(drop=True)
        nms_dets.to_csv(f'{SAVE_PATH}{seq_name}.csv', sep=" ", index=False)