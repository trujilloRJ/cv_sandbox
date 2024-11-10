import cv2 as cv
import pandas as pd
import numpy as np
from os import listdir
from utils import KEY_A, KEY_ESC, KEY_D, KEY_M, KEY_N, draw_detection_bb, draw_track_bb
from configuration import get_config


SEQUENCE = "0002"
IMG_PATH = f"data/{SEQUENCE}/"
DET_FILE = f"data/nms/{SEQUENCE}.csv"
TRACK_FILE = f"data/tracks/{SEQUENCE}.csv"
DATASET = 'KITTI'

def load_img(frame_name, dets, show_dets, show_tracks):
    frame_index = int(frame_name[:-4])
    frame = cv.imread(IMG_PATH + frame_name, cv.IMREAD_COLOR)
    cur_dets = dets.loc[dets['frame'] == frame_index, :]

    # drawing
    if show_dets:
        for det_id in cur_dets.index:
            det = cur_dets.loc[det_id, :]
            draw_detection_bb(frame, det)

    if show_tracks:
        # cur_tracks = tracks.loc[(tracks['frame'] == frame_index) & (tracks['tentative'] == False), :]
        cur_tracks = tracks.loc[(tracks['frame'] == frame_index), :]
        # drawing track bounding box
        for _, track in cur_tracks.iterrows():
            track_bb = [int(track['top']), int(track['left']), int(track['bottom']), int(track['right'])]
            draw_track_bb(frame, track['id'], track['type'], track_bb)

    cv.putText(frame, f'{frame_name}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

    return frame


if __name__ == '__main__':
    
    show_dets = True
    show_tracks = True

    config = get_config(DATASET)
    frames = listdir(IMG_PATH)

    dets = pd.read_csv(DET_FILE, header=None, sep=" ")
    dets.columns = config['det_cols']

    tracks = pd.read_csv(TRACK_FILE)
    cols_to_round = ['left', 'top', 'right', 'bottom']
    tracks[cols_to_round] = np.round(tracks[cols_to_round]).astype(int)

    cur_frame = 0
    run = True

    while(run):
        frame_name = frames[cur_frame]
        frame = load_img(frame_name, dets, show_dets, show_tracks)

        cv.imshow("display", frame)
        key = cv.waitKey(0)

        if key == KEY_D:
            cur_frame += 1
        if key == KEY_A:
            cur_frame -= 1
        if key == KEY_N:
            show_dets = not show_dets
        if key == KEY_M:
            show_tracks = not show_tracks
        if key == KEY_ESC:
            run = False
