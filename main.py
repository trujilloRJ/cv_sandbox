import cv2 as cv
import pandas as pd
import numpy as np
from os import listdir
from tracks import Track
from association import compute_iou
from scipy.optimize import linear_sum_assignment

IMG_PATH = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/img1/"
DET_FILE = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/det/det.txt"
COLS = ['frame', 'track', 'bb_left', 'bb_top', 'width', 'height', 'confidence', 'x', 'y', 'z']


def create_new_track(track_list, det, cycle_time):
    id_ = 0 if len(track_list) == 0 else track_list[-1].id + 1
    return Track(id_, det['center_x'].item(), det['center_y'].item(), det['width'].item(), det['height'].item(), cycle_time)


def do_track_cyclic(track_list, dets, cycle, cycle_time):
    if cycle == 0:
        # create tracks from detections
        for det_id in cur_dets.index:
            det = pd.Series(cur_dets.loc[det_id, :])
            new_trk = create_new_track(track_list, det, cycle_time)
            track_list.append(new_trk)
    else:
        n_dets, n_tracks = len(dets), len(track_list)

        # predict tracks
        for trk in track_list:
            trk.predict()

        # associate
        cost_thr = 0.2 # minimum IoU cost to declare a match
        cost_matrix = np.zeros((n_tracks, n_dets), dtype=float)
        for i, track in enumerate(track_list):
            for j, det_id in enumerate(cur_dets.index):
                det = pd.Series(cur_dets.loc[det_id, :])
                det_bb = np.array([det['bb_top'].item(), det['bb_left'].item(), det['bb_bottom'].item(), det['bb_right'].item()])
                cost_matrix[i, j] = compute_iou(track.bb, det_bb)
        track_indices, det_indices = linear_sum_assignment(cost_matrix, maximize=True)
        matched_tracks_id = []
        matched_dets_id = []
        for i, j in zip(track_indices, det_indices):
            if cost_matrix[i, j] > cost_thr:
                matched_tracks_id.append(i.item())
                matched_dets_id.append(cur_dets.index[j].item())

        # update associated tracks and track maintenance
        tracks_ids_to_delete = []
        for i, trk in enumerate(track_list):
            trk.life_count += 1
            if i in matched_tracks_id:
                trk.match = True
                trk.unmatch_count = 0

                det_index = matched_dets_id[matched_tracks_id.index(i)]
                det = pd.Series(cur_dets.loc[det_index, :])
                meas = np.array([det['center_x'].item(), det['center_y'].item(), det['width'].item(), det['height'].item()])
                trk.update(meas)
            else:
                trk.match = False
                trk.unmatch_count += 1
                if trk.unmatch_count >= 5:
                    tracks_ids_to_delete.append(trk.id)


        # new tracks from unassociated detections
        for det_id in cur_dets.index:
            if det_id not in matched_dets_id:
                det = pd.Series(cur_dets.loc[det_id, :])
                new_trk = create_new_track(track_list,det, cycle_time)
                track_list.append(new_trk)

        # delete tracks
        track_list = [trk for trk in track_list if trk.id not in tracks_ids_to_delete]


if __name__ == "__main__":
    # load detections
    dets = pd.read_csv(DET_FILE, header=None)
    fps = 30
    dets.columns = COLS
    cols_to_round = ['bb_left', 'bb_top', 'height', 'width']
    dets[cols_to_round] = np.round(dets[cols_to_round]).astype(int)
    frames = listdir(IMG_PATH)
    track_list = []
    cycle_time = 1/fps

    # pre-process dets
    dets.loc[:, 'center_x'] = (dets.loc[:, 'bb_top'] + dets.loc[:, 'height'] // 2).astype(int)
    dets.loc[:, 'center_y'] = (dets.loc[:, 'bb_left'] + dets.loc[:, 'width'] // 2).astype(int)
    dets.loc[:, 'bb_bottom'] = (dets.loc[:, 'bb_top'] + dets.loc[:, 'height']).astype(int)
    dets.loc[:, 'bb_right'] = (dets.loc[:, 'bb_left'] + dets.loc[:, 'width']).astype(int)
    
    for cyc, frame_name in enumerate(frames):

        # load detections
        frame_index = int(frame_name[:-4])
        frame = cv.imread(IMG_PATH + frame_name, cv.IMREAD_COLOR)
        cur_dets = dets.loc[dets['frame'] == frame_index, :]

        # tracking cycle here
        do_track_cyclic(track_list, cur_dets, cyc, cycle_time)

        # drawing det bounding box
        for det_id in cur_dets.index:
            det = cur_dets.loc[det_id, :]
            bbtl = (int(det['bb_left']), int(det['bb_top']))
            bbrb = (int(bbtl[0] + det['width']), int(bbtl[1] + det['height']))
            cv.rectangle(frame, bbtl, bbrb, (0,255,0), 1)
            # cv.circle(frame, (int(det['center_y']), int(det['center_x'])), 5, (0, 0, 255), -1)

        # drawing track bounding box
        for track in track_list:
            cv.rectangle(frame, (track.bb[1], track.bb[0]), (track.bb[3], track.bb[2]), (0, 0, 255), 1)
            cv.putText(frame, f'{track.id}', (track.bb[1]-10, track.bb[0]-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        
        # show image
        cv.imshow("display", frame)
        cv.waitKey(0)

    cv.destroyAllWindows()