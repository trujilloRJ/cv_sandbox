import cv2 as cv
import numpy as np
from association import compute_iou

def draw_box_and_iou(img, box1, box2):
    cv.rectangle(img, (box1[0], box1[1]), (box1[2], box1[3]), (0, 0, 255), thickness=1)
    cv.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), (0, 255, 0), thickness=1)
    iou = compute_iou(box1, box2)
    cv.putText(img, f'IoU: {round(iou, 2)}', (box1[0]-10, box1[1]-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))


if __name__ == '__main__':
    blank = np.zeros((500, 500, 3))

    box1 = np.array([100, 100, 200, 200])
    box2 = np.array([150, 150, 200, 200])
    draw_box_and_iou(blank, box1, box2)

    box2 = np.array([300, 300, 350, 350])
    box1 = np.array([375, 375, 400, 400])
    draw_box_and_iou(blank, box1, box2)

    box2 = np.array([100, 300, 200, 350])
    box1 = np.array([102, 302, 202, 352])
    draw_box_and_iou(blank, box1, box2)

    cv.imshow('test_iou', blank)
    cv.waitKey(0)
    cv.destroyAllWindows()