import numpy as np 


def compute_iou(box1: np.ndarray, box2: np.ndarray):
    a1, b1, c1, d1 = box1  # top, left, bottom, right
    a2, b2, c2, d2 = box2

    # getting intersection indices
    ai = max(a1, a2)
    bi = max(b1, b2)
    ci = min(c1, c2)
    di = min(d1, d2)

    # no need to do additional calculations if there is no intersection
    if ci - ai <= 0:
        return 0
    
    area_inter = (ci - ai)*(di - bi)

    # computing union area
    area1 = np.abs((a1 - c1) * (b1 - d1))
    area2 = np.abs((a2 - c2) * (b2 - d2))
    area_union = area1 + area2 - area_inter

    return area_inter/area_union
