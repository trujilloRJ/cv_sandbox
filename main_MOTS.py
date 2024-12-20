import cv2 as cv
import pandas as pd
import numpy as np
from os import listdir
from track_main import do_track_cyclic
from utils import KEY_ESC

IMG_PATH = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/img1/"
DET_FILE = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/det/det.txt"
COLS = ['frame', 'track', 'bb_left', 'bb_top', 'width', 'height', 'confidence', 'x', 'y', 'z']
SHOW_WINDOW = True


if __name__ == "__main__":
    # load detections
    fps = 30
    cycle_time = 1/fps
    dets = pd.read_csv(DET_FILE, header=None)
    dets.columns = COLS
    cols_to_round = ['bb_left', 'bb_top', 'height', 'width']
    dets[cols_to_round] = np.round(dets[cols_to_round]).astype(int)
    frames = listdir(IMG_PATH)
    global_track_list = []
    track_list = []

    # pre-process detections
    dets.loc[:, 'center_x'] = (dets.loc[:, 'bb_top'] + dets.loc[:, 'height'] // 2).astype(int)
    dets.loc[:, 'center_y'] = (dets.loc[:, 'bb_left'] + dets.loc[:, 'width'] // 2).astype(int)
    dets.loc[:, 'bb_bottom'] = (dets.loc[:, 'bb_top'] + dets.loc[:, 'height']).astype(int)
    dets.loc[:, 'bb_right'] = (dets.loc[:, 'bb_left'] + dets.loc[:, 'width']).astype(int)
    
    for cyc, frame_name in enumerate(frames):
        print(cyc)

        # load detections
        frame_index = int(frame_name[:-4])
        frame = cv.imread(IMG_PATH + frame_name, cv.IMREAD_COLOR)
        cur_dets = dets.loc[dets['frame'] == frame_index, :]

        # tracking cycle here
        track_list = do_track_cyclic(track_list, cur_dets, cyc, cycle_time)
        for trk in track_list:
            trk_dict = trk.to_dict()
            trk_dict.update({'frame': cyc})
            global_track_list.append(trk_dict)

        if SHOW_WINDOW:
            # drawing det bounding box
            for det_id in cur_dets.index:
                det = cur_dets.loc[det_id, :]
                bbtl = (int(det['bb_left']), int(det['bb_top']))
                bbrb = (int(bbtl[0] + det['width']), int(bbtl[1] + det['height']))
                cv.rectangle(frame, bbtl, bbrb, (0,255,0), 1)
                # cv.circle(frame, (int(det['center_y']), int(det['center_x'])), 5, (0, 0, 255), -1)

            # drawing track bounding box
            for track in track_list:
                if not track.tentative:
                    cv.rectangle(frame, (track.bb[1], track.bb[0]), (track.bb[3], track.bb[2]), (0, 0, 255), 1)
                    cv.putText(frame, f'{track.id}', (track.bb[1]-10, track.bb[0]-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            cv.putText(frame, f'{cyc}:{frame_name}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
            
            # show image
            cv.imshow("display", frame)
            key = cv.waitKey(0)
            if key == KEY_ESC:
                break

    # save tracks
    tracks = pd.DataFrame.from_records(global_track_list)
    tracks.to_csv("tracks.csv")

    cv.destroyAllWindows()