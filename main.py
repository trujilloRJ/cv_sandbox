import cv2 as cv
import pandas as pd
import numpy as np
from os import listdir
from tracks import Track

IMG_PATH = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/img1/"
DET_FILE = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/det/det.txt"
COLS = ['frame', 'track', 'bb_left', 'bb_top', 'height', 'width', 'confidence', 'class', 'occlusion', 'dummy']


def do_track_cyclic(track_list, dets, cycle, cycle_time):
    if cycle == 0:
        # create tracks from detections
        for det_id in cur_dets.index:
            det = pd.Series(cur_dets.loc[det_id, :])
            id_ = len(track_list) + 1
            new_trk = Track(id_, det['center_x'].item(), det['center_y'].item(), det['width'].item(), det['height'].item(), cycle_time)
            track_list.append(new_trk)
    else:
        # predict tracks
        for trk in track_list:
            trk.predict()

        # associate

        # update associated tracks

        # new tracks from unassociated detections

        # track maintenance

    


if __name__ == "__main__":
    # load detections
    dets = pd.read_csv(DET_FILE, header=None)
    fps = 20
    dets.columns = COLS
    cols_to_round = ['bb_left', 'bb_top', 'height', 'width']
    dets[cols_to_round] = np.round(dets[cols_to_round]).astype(int)
    frames = listdir(IMG_PATH)
    track_list = []
    cycle_time = 1/fps

    # pre-process dets
    dets.loc[:, 'center_x'] = (dets.loc[:, 'bb_top'] + dets.loc[:, 'width'] // 2).astype(int)
    dets.loc[:, 'center_y'] = (dets.loc[:, 'bb_left'] + dets.loc[:, 'height'] // 2).astype(int)
    
    for cyc, frame_name in enumerate(frames):

        # load detections
        frame_index = int(frame_name[:-4])
        frame = cv.imread(IMG_PATH + frame_name, cv.IMREAD_COLOR)
        cur_dets = dets.loc[dets['frame'] == frame_index, :]

        # tracking cycle here
        # do_track_cyclic(track_list, cur_dets, cyc, cycle_time)

        # drawing bounding box
        for det_id in cur_dets.index:
            det = cur_dets.loc[det_id, :]
            bbtl = (int(det['bb_left']), int(det['bb_top']))
            bbrb = (int(bbtl[0] + det['height']), int(bbtl[1] + det['width']))
            cv.rectangle(frame, bbtl, bbrb, (0,255,0), 1)
            cv.circle(frame, (int(det['center_y']), int(det['center_x'])), 5, (0, 0, 255), -1)
        
        # show image
        cv.imshow("display", frame)
        cv.waitKey(0)

    cv.destroyAllWindows()