import cv2 as cv
import pandas as pd
import numpy as np
from os import listdir
from utils import KEY_A, KEY_ESC, KEY_D, KEY_M, KEY_N, KEY_G, draw_detection_bb, draw_track_bb, draw_gt_bb, mark_gt
from configuration import get_config

TRACKER_NAME = 'customSORT_v1_newScore'
SEQUENCE = "0002"
IMG_PATH = f"data/images/{SEQUENCE}/"
DET_FILE = f"data/nms/{SEQUENCE}.csv"
# TRACK_FILE = f'data/tracks/{TRACKER_NAME}/data/{SEQUENCE}.txt'
TRACK_FILE = f'data/tracks/{SEQUENCE}.txt'
GT_FILE = f"data/gt/{SEQUENCE}.txt"
DATASET = 'KITTI'

def load_img(frame_name, dets, tracks, gt, show_dets, show_tracks, show_gt):
    frame_index = int(frame_name[:-4])
    frame = cv.imread(IMG_PATH + frame_name, cv.IMREAD_COLOR)

    cur_dets = dets.loc[dets['frame'] == frame_index, :]
    cur_tracks = tracks.loc[(tracks['frame'] == frame_index), :]
    cur_gt = gt.loc[(gt['frame'] == frame_index), :]

    # drawing dets
    if show_dets:
        for det_id in cur_dets.index:
            det = cur_dets.loc[det_id, :]
            draw_detection_bb(frame, det)

    # drawing track bounding box
    if show_tracks:
        for _, track in cur_tracks.iterrows():
            track_bb = [int(track['top']), int(track['left']), int(track['bottom']), int(track['right'])]
            draw_track_bb(frame, track['id'], track['type'], track['score'], track_bb)

    # drawing track bounding box
    if show_gt:
        frame = draw_filled_gt(frame, cur_gt)

    cv.putText(frame, f'{frame_name}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

    return frame


def draw_filled_gt(frame, cur_gt):
    shapes = np.zeros_like(frame, np.uint8)
    for _, gt_ in cur_gt.iterrows():
        bb = [int(gt_['top']), int(gt_['left']), int(gt_['bottom']), int(gt_['right'])]
        draw_gt_bb(shapes, bb)
        mark_gt(frame, gt_['id'], gt_['type'], bb)
    alpha = 0.8
    mask = shapes.astype(bool)
    frame[mask] = cv.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
    return frame

def click_callback(event, x, y, *args):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)


if __name__ == '__main__':
    
    show_dets = False
    show_tracks = True
    show_gt = True

    config = get_config(DATASET)
    frames = listdir(IMG_PATH)

    dets = pd.read_csv(DET_FILE, sep=' ')

    tracks = pd.read_csv(TRACK_FILE, sep=' ')
    tracks.columns = config['track_cols']
    cols_to_round = ['left', 'top', 'right', 'bottom']
    tracks[cols_to_round] = np.round(tracks[cols_to_round]).astype(int)

    gt = pd.read_csv(GT_FILE, sep=' ')
    gt.columns = config['track_cols'][:-1]
    gt[cols_to_round] = np.round(gt[cols_to_round]).astype(int)
    gt = gt.loc[gt['type'] != 'DontCare', :]

    cur_frame = 0
    run = True

    cv.namedWindow("display")
    cv.setMouseCallback("display", click_callback)

    while(run):
        frame_name = frames[cur_frame]
        frame = load_img(frame_name, dets, tracks, gt, show_dets, show_tracks, show_gt)

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
        if key == KEY_G:
            show_gt = not show_gt
        if key == KEY_ESC:
            run = False
        else:
            print(key)
