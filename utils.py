import random
import cv2 as cv

OBJ_TYPES = {'Car': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist'}

KEY_ESC = 27
KEY_SPACE = 32
KEY_A = 97
KEY_D = 100
KEY_N = 110
KEY_M = 109
KEY_G = 103
KEY_S = 115

NUM_COLORS = 15
COLOR_DET = {
    OBJ_TYPES['Car'] : (0,255,0),
    OBJ_TYPES['Pedestrian'] : (255,0,0),
    OBJ_TYPES['Cyclist'] : (255,255,0),
}
COLOR_TRACK = (0, 0, 255)
COLOR_GT = (0, 255, 0)

COLOR_LIST = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), 
    (70, 240, 240), (240, 50, 230), (250, 190, 212), (0, 128, 128), (220, 190, 255), 
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (0, 0, 128)]

def draw_gt_bb(frame, gt_bb):
    cv.rectangle(frame, (gt_bb[1], gt_bb[0]), (gt_bb[3], gt_bb[2]), COLOR_GT, cv.FILLED)

def mark_gt(frame, gt_id, gt_type, gt_bb):
    cv.putText(frame, f'{gt_id}:{gt_type}', (gt_bb[1] + 50, int((gt_bb[0] + gt_bb[2])/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GT)

def draw_track_bb(frame, track_id, track_type, track_score, track_bb):
    color = COLOR_LIST[track_id % NUM_COLORS]
    cv.rectangle(frame, (track_bb[1], track_bb[0]), (track_bb[3], track_bb[2]), color, 2)

    # Draw the filled rectangle (background)
    text_size, _ = cv.getTextSize(f'{track_id}', cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_width, text_height = text_size
    padding = 2
    cv.rectangle(frame, (track_bb[1]-padding, track_bb[0]-padding), (track_bb[1]+text_width+padding, track_bb[0]-text_height-padding), (255, 255, 255), -1)
    cv.putText(frame, f'{track_id}', (track_bb[1], track_bb[0]-3), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_detection_bb(frame, det):
    bbtl = (int(det['left']), int(det['top']))
    bbrb = (int(det['right']), int(det['bottom']))
    cv.rectangle(frame, bbtl, bbrb, COLOR_DET[det['type']], 1)

