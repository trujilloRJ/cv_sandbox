import cv2 as cv
import pandas as pd
import numpy as np
from os import listdir
from utils import KEY_A, KEY_ESC, KEY_D


IMG_PATH = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/img1/"
DET_FILE = r"C:/javier/personal_projects/computer_vision/data/MOT16/train/MOT16-02/det/det.txt"
TRACK_FILE = r"C:/javier/personal_projects/computer_vision/cv_sandbox/tracks.csv"
COLS = ['frame', 'track', 'bb_left', 'bb_top', 'width', 'height', 'confidence', 'x', 'y', 'z']


def load_img(frame_name, dets, tracks):
    frame_index = int(frame_name[:-4])
    frame = cv.imread(IMG_PATH + frame_name, cv.IMREAD_COLOR)
    cur_dets = dets.loc[dets['frame'] == frame_index, :]
    cur_tracks = tracks.loc[tracks['frame'] == frame_index, :]

    # drawing det bounding box
    for det_id in cur_dets.index:
        det = cur_dets.loc[det_id, :]
        bbtl = (int(det['bb_left']), int(det['bb_top']))
        bbrb = (int(bbtl[0] + det['width']), int(bbtl[1] + det['height']))
        cv.rectangle(frame, bbtl, bbrb, (0,255,0), 1)
        # cv.circle(frame, (int(det['center_y']), int(det['center_x'])), 5, (0, 0, 255), -1)

    # drawing track bounding box
    for track_id, track in cur_tracks.iterrows():
        cv.rectangle(frame, (int(track['left']), int(track['top'])), (int(track['right']), int(track['bottom'])), (0, 0, 255), 1)
        cv.putText(frame, f'{track['id']}', (int(track['left'])-10, int(track['top'])-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv.putText(frame, f'{frame_name}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

    return frame


if __name__ == '__main__':
    frames = listdir(IMG_PATH)

    dets = pd.read_csv(DET_FILE, header=None)
    dets.columns = COLS
    cols_to_round = ['bb_left', 'bb_top', 'height', 'width']
    dets[cols_to_round] = np.round(dets[cols_to_round]).astype(int)

    tracks = pd.read_csv(TRACK_FILE)
    cols_to_round = ['left', 'top', 'right', 'bottom']
    tracks[cols_to_round] = np.round(tracks[cols_to_round]).astype(int)

    cur_frame = 0
    run = True

    while(run):
        frame_name = frames[cur_frame]
        frame = load_img(frame_name, dets, tracks)

        cv.imshow("display", frame)
        key = cv.waitKey(0)

        if key == KEY_D:
            cur_frame += 1
        if key == KEY_A:
            cur_frame -= 1
        if key == KEY_ESC:
            run = False
