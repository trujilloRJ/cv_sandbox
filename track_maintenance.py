import numpy as np
from tracks import Track


SCORE_LLR_MIN = -10
SCORE_LLR_MAX = 10
SCORE_IOU = 'IOU'
SCORE_NIS = 'NIS'


def create_new_track(track_list, det, cycle_time):
    id_ = 0 if len(track_list) == 0 else track_list[-1].id + 1
    new_trk = Track(
        id_, det['center_x'].item(), det['center_y'].item(), 
        det['width'].item(), det['height'].item(), 
        cycle_time, det['type']
    )
    return new_trk


def update_track_score(track: Track, score_fn: str, *args):
    if score_fn == SCORE_IOU:
        delta_llr = update_score_iou(track, *args)
    elif score_fn == SCORE_NIS:
        delta_llr = update_score_nis(track, *args)
    else:
        raise ValueError(f'{score_fn} is an invalid score function')
    
    # putting limits to log-likelihood ratio to be more sensible during long tracks
    track.score_llr = max(min(track.score_llr + delta_llr, SCORE_LLR_MAX), SCORE_LLR_MIN)

def update_score_iou(track: Track, iou, cf, Pd, Pfa):
    if track.match:
        delta_llr = np.log(iou) - np.log(cf)
    else:
        delta_llr = np.log(1 - Pd) - np.log(1 - Pfa)
    return delta_llr

def update_score_nis(track, Pd, Bft):
    if track.match:
        S = track.KF.S
        dim_z = S.shape[0]
        delta_llr = np.log(Pd/((2*np.pi)**(dim_z/2) * Bft * np.sqrt(np.linalg.det(S)))) - track.KF.nis/2
    else:
        delta_llr = np.log(1 - Pd)
    return delta_llr