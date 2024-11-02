import numpy as np
import pandas as pd 
from scipy.optimize import linear_sum_assignment

COST_THR = 0.2 # minimum IoU cost to declare a match

def associate_tracks_dets(track_list, cur_dets):
    matched_tracks_id = []
    matched_dets_id = []
    n_dets, n_tracks = len(cur_dets), len(track_list)
    cost_matrix = np.zeros((n_tracks, n_dets), dtype=float)
    for i, track in enumerate(track_list):
        for j, det_id in enumerate(cur_dets.index):
            det = pd.Series(cur_dets.loc[det_id, :])
            det_bb = np.array([det['bb_top'].item(), det['bb_left'].item(), det['bb_bottom'].item(), det['bb_right'].item()])
            cost_matrix[i, j] = compute_iou(track.bb, det_bb)

    track_indices, det_indices = linear_sum_assignment(cost_matrix, maximize=True)
    for i, j in zip(track_indices, det_indices):
        if cost_matrix[i, j] > COST_THR:
            matched_tracks_id.append(i.item())
            matched_dets_id.append(cur_dets.index[j].item())
    
    return matched_tracks_id, matched_dets_id


def compute_iou(box1: np.ndarray, box2: np.ndarray):
    a1, b1, c1, d1 = box1  # top, left, bottom, right
    a2, b2, c2, d2 = box2

    # getting intersection indices
    ai = max(a1, a2)
    bi = max(b1, b2)
    ci = min(c1, c2)
    di = min(d1, d2)

    # no need to do additional calculations if there is no intersection
    if ci - ai <= 0:
        return 0
    
    area_inter = (ci - ai)*(di - bi)

    # computing union area
    area1 = np.abs((a1 - c1) * (b1 - d1))
    area2 = np.abs((a2 - c2) * (b2 - d2))
    area_union = area1 + area2 - area_inter

    return area_inter/area_union
