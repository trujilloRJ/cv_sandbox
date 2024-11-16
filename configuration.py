KITTI = {
    'det_cols': ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'left', 'top', 'right', 'bottom', 'height', 'width', 'length', 'x', 'y', 'z', 'rotation_y', 'score'],
    'track_cols': ['frame', 'id', 'type', 'truncated', 'occluded', 'alpha', 
                  'left', 'top', 'right', 'bottom', 
                  'height_m', 'width_m', 'length_m',
                  'x', 'y', 'z', 'rotation_y', 'score'],
    'FPS': 10
}

MOTS = {
    'det_cols': ['frame', 'track', 'bb_left', 'bb_top', 'width', 'height', 'confidence', 'x', 'y', 'z']
}

def get_config(dataset):
    if dataset == 'KITTI':
        return KITTI
    elif dataset == 'MOTS':
        return MOTS 