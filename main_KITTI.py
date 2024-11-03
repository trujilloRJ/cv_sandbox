import cv2 as cv
import pandas as pd
import numpy as np
from os import listdir
from track_main import do_track_cyclic
from utils import KEY_ESC
from configuration import get_config

IMG_PATH = r"C:/javier/personal_projects/computer_vision/data/KITTI_object_tracking/images/training/image_02/0000/"
DET_FILE = r"C:/javier/personal_projects/computer_vision/data/KITTI_object_tracking/detections_regionlet/training/det_02/0000.txt"
DATASET = 'KITTI'
SHOW_WINDOW = True


if __name__ == "__main__":
    config = get_config(DATASET)
    fps = config['FPS']
    cycle_time = 1/fps

    # load detections
    dets = pd.read_csv(DET_FILE, header=None, sep=' ')
    dets.columns = config['det_cols']
    frames = listdir(IMG_PATH)
    global_track_list = []
    track_list = []

    # pre-process detections
    dets = dets.loc[dets['score'] > 0, :]
    dets.loc[:, 'center_x'] = (dets.loc[:, 'top'] + dets.loc[:, 'bottom']) // 2
    dets.loc[:, 'center_y'] = (dets.loc[:, 'left'] + dets.loc[:, 'right']) // 2
    dets.loc[:, 'width'] = (dets.loc[:, 'right'] - dets.loc[:, 'left']).astype(int)
    dets.loc[:, 'height'] = (dets.loc[:, 'bottom'] - dets.loc[:, 'top']).astype(int)
    
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
                bbtl = (int(det['left']), int(det['top']))
                bbrb = (int(det['right']), int(det['bottom']))
                cv.rectangle(frame, bbtl, bbrb, (0,255,0), 1)

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