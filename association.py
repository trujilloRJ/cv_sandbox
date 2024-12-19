import numpy as np
import pandas as pd 
from scipy.optimize import linear_sum_assignment

COST_MIN = -9999.0  # minimum cost in iou terms to rule out detection
COST_THR = 0.2 # minimum IoU cost to declare a match

def associate_main(track_list, cur_dets):
    primary_track_list = [trk for trk in track_list if trk.tentative == False]
    tentative_track_list = [trk for trk in track_list if trk.tentative == True]

    # associating primary tracks first
    matched_ptracks_id, matched_pdets_id, p_assoc_scores = associate_tracks_dets(primary_track_list, cur_dets)
    cur_unmantched_dets = cur_dets.loc[~cur_dets.index.isin(matched_pdets_id), :]
    matched_ttracks_id, matched_tdets_id, t_assoc_scores = associate_tracks_dets(tentative_track_list, cur_unmantched_dets)
    
    matched_tracks_id = matched_ptracks_id + matched_ttracks_id
    matched_dets_id = matched_pdets_id + matched_tdets_id
    assoc_scores = p_assoc_scores + t_assoc_scores

    return matched_tracks_id, matched_dets_id, assoc_scores


def associate_tracks_dets(track_list, cur_dets):
    matched_tracks_id = []
    matched_dets_id = []
    assoc_scores = []     

    cost_matrix = compute_cost_matrix(track_list, cur_dets)
    track_indices, det_indices = linear_sum_assignment(cost_matrix, maximize=True)

    # filling match tracks and detection lists
    for i, j in zip(track_indices, det_indices):
        if cost_matrix[i, j] > COST_THR:
            matched_tracks_id.append(track_list[i.item()].id)
            matched_dets_id.append(cur_dets.index[j].item())
            assoc_scores.append(cost_matrix[i, j])
    
    return matched_tracks_id, matched_dets_id, assoc_scores


def compute_cost_matrix(track_list, dets: pd.DataFrame):
    allowed_class_mismatch = ['Pedestrian', 'Cyclist']
    n_tracks, n_dets = len(track_list), len(dets)
    cost_matrix = np.full((n_tracks, n_dets), COST_MIN, dtype=float)
    n_dets, n_tracks = len(dets), len(track_list)
    for i, track in enumerate(track_list):
        for j, det_id in enumerate(dets.index):
            det = pd.Series(dets.loc[det_id, :])
            det_type = det['type']
            if det_type == track.type or (det_type in allowed_class_mismatch and track.type in allowed_class_mismatch):
                det_bb = np.array([det['top'].item(), det['left'].item(), det['bottom'].item(), det['right'].item()])
                cost_matrix[i, j] = compute_iou(track.bb, det_bb)
    return cost_matrix

def compute_iou(box1: np.ndarray, box2: np.ndarray):
    a1, b1, c1, d1 = box1  # top, left, bottom, right
    a2, b2, c2, d2 = box2

    # getting intersection indices
    ai = max(a1, a2)
    bi = max(b1, b2)
    ci = min(c1, c2)
    di = min(d1, d2)

    # no need to do additional calculations if there is no intersection
    if (ci - ai <= 0) or (di - bi <= 0):
        return COST_MIN
    
    area_inter = (ci - ai)*(di - bi)

    # computing union area
    area1 = np.abs((a1 - c1) * (b1 - d1))
    area2 = np.abs((a2 - c2) * (b2 - d2))
    area_union = area1 + area2 - area_inter

    return area_inter/area_union
