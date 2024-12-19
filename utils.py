import cv2 as cv

OBJ_TYPES = {'Car': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist'}

KEY_ESC = 27
KEY_SPACE = 32
KEY_A = 97
KEY_D = 100
KEY_N = 110
KEY_M = 109
KEY_G = 103

COLOR_DET = {
    OBJ_TYPES['Car'] : (0,255,0),
    OBJ_TYPES['Pedestrian'] : (255,0,0),
    OBJ_TYPES['Cyclist'] : (255,255,0),
}
COLOR_TRACK = (0, 0, 255)
COLOR_GT = (0, 255, 0)

def draw_gt_bb(frame, gt_bb):
    cv.rectangle(frame, (gt_bb[1], gt_bb[0]), (gt_bb[3], gt_bb[2]), COLOR_GT, cv.FILLED)

def mark_gt(frame, gt_id, gt_type, gt_bb):
    cv.putText(frame, f'{gt_id}:{gt_type}', (gt_bb[1] + 50, int((gt_bb[0] + gt_bb[2])/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GT)

def draw_track_bb(frame, track_id, track_type, track_score, track_bb):
    cv.rectangle(frame, (track_bb[1], track_bb[0]), (track_bb[3], track_bb[2]), COLOR_TRACK, 1)
    cv.putText(frame, f'{track_id}:{round(track_score, 2)}', (track_bb[1]-10, track_bb[0]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TRACK)


def draw_detection_bb(frame, det):
    bbtl = (int(det['left']), int(det['top']))
    bbrb = (int(det['right']), int(det['bottom']))
    cv.rectangle(frame, bbtl, bbrb, COLOR_DET[det['type']], 1)
