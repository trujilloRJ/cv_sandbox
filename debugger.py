import cv2 as cv
import pandas as pd
import numpy as np
from os import listdir
from utils import KEY_A, KEY_ESC, KEY_D, KEY_M, KEY_N
from configuration import get_config


IMG_PATH = r"C:/javier/personal_projects/computer_vision/data/KITTI_object_tracking/images/training/image_02/0000/"
SEQUENCE = "0000"
# DET_FILE = f"C:/javier/personal_projects/computer_vision/data/KITTI_object_tracking/detections_regionlet/training/det_02/{SEQUENCE}.txt"
DET_FILE = f"C:/javier/personal_projects/computer_vision/cv_sandbox/{SEQUENCE}_nms_dets.csv"
DATASET = 'KITTI'

def load_img(frame_name, dets, show_dets, show_tracks):
    frame_index = int(frame_name[:-4])
    frame = cv.imread(IMG_PATH + frame_name, cv.IMREAD_COLOR)
    cur_dets = dets.loc[dets['frame'] == frame_index, :]

    # drawing det bounding box
    if show_dets:
        for det_id in cur_dets.index:
            det = cur_dets.loc[det_id, :]
            bbtl = (int(det['left']), int(det['top']))
            bbrb = (int(det['right']), int(det['bottom']))
            cv.rectangle(frame, bbtl, bbrb, (0,255,0), 1)

    # if show_tracks:
    # cur_tracks = tracks.loc[tracks['frame'] == frame_index, :]
    #     # drawing track bounding box
    #     for track_id, track in cur_tracks.iterrows():
    #         cv.rectangle(frame, (int(track['left']), int(track['top'])), (int(track['right']), int(track['bottom'])), (0, 0, 255), 1)
    #         cv.putText(frame, f'{int(track['id'])}', (int(track['left'])-10, int(track['top'])-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv.putText(frame, f'{frame_name}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

    return frame


if __name__ == '__main__':
    
    show_dets = True
    show_tracks = False

    config = get_config(DATASET)
    frames = listdir(IMG_PATH)

    dets = pd.read_csv(DET_FILE, header=None, sep=" ")
    dets.columns = config['det_cols']
    dets = dets.loc[dets['score'] > 0, :]

    # tracks = pd.read_csv(TRACK_FILE)
    # cols_to_round = ['left', 'top', 'right', 'bottom']
    # tracks[cols_to_round] = np.round(tracks[cols_to_round]).astype(int)

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
