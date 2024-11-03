from tracks import Track

def create_new_track(track_list, det, cycle_time):
    id_ = 0 if len(track_list) == 0 else track_list[-1].id + 1
    return Track(id_, det['center_x'].item(), det['center_y'].item(), det['width'].item(), det['height'].item(), cycle_time)