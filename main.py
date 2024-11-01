import cv2 as cv
import pandas as pd
import numpy as np
from os import listdir

IMG_PATH = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/img1/"
DET_FILE = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/det/det.txt"
COLS = ['frame', 'track', 'bbtl_y', 'bbtl_x', 'height', 'width', 'confidence', 'class', 'occlusion', 'dummy']

if __name__ == "__main__":
    # load detections
    dets = pd.read_csv(DET_FILE, header=None)
    dets.columns = COLS
    cols_to_round = ['bbtl_y', 'bbtl_x', 'height', 'width']
    dets[cols_to_round] = np.round(dets[cols_to_round]).astype(int)
    frames = listdir(IMG_PATH)
    
    for frame_name in frames:
        frame_index = int(frame_name[:-4])
        frame = cv.imread(IMG_PATH + frame_name, cv.IMREAD_COLOR)

        cur_dets = dets.loc[dets['frame'] == frame_index, :]
        # drawing bounding box
        for det_id in cur_dets.index:
            det = cur_dets.loc[det_id, :]
            bbtl = (int(det['bbtl_y']), int(det['bbtl_x']))
            bbrb = (int(bbtl[0] + det['height']), int(bbtl[1] + det['width']))
            cv.rectangle(frame, bbtl, bbrb, (0,255,0), 1)
        
        # show image
        cv.imshow("display", frame)
        cv.waitKey(0)

    cv.destroyAllWindows()