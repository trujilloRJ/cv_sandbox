import pandas as pd 
import numpy as np
from typing import List
from track_maintenance import create_new_track
from association import associate_main, associate_tracks_dets
from configuration import get_config
from tracks import Track


def main_cyclic(det_fn: str, dataset = 'KITTI', out_tentative = True):
    config = get_config(dataset)
    fps = config['FPS']
    cycle_time = 1/fps
    global_track_list = []
    track_list = []
    
    # load detections
    dets = pd.read_csv(det_fn, sep=' ')
    # frames = np.unique(dets['frame'])
    frames = np.arange(np.min(dets['frame']), np.max(dets['frame']))

    for cyc, frame_index in enumerate(frames):
        # filter detections
        cur_dets = dets.loc[dets['frame'] == frame_index, :]

        # tracking cycle here
        track_list = do_track_cyclic(track_list, cur_dets, cyc, cycle_time)

        # save tracks in global list
        for trk in track_list:
            if trk.tentative and not out_tentative:
                continue
            trk_dict = trk.to_dict()
            trk_dict.update({'frame': cyc})
            global_track_list.append(trk_dict)

    # format tracks
    tracks_df = pd.DataFrame.from_records(global_track_list)
    tracks_df = format_tracks_for_eval(tracks_df)

    return tracks_df


def do_track_cyclic(track_list: List[Track], cur_dets, cycle, cycle_time):
    if cycle == 0:
        # create tracks from detections
        for det_id in cur_dets.index:
            det = pd.Series(cur_dets.loc[det_id, :])
            new_trk = create_new_track(track_list, det, cycle_time)
            track_list.append(new_trk)
    else:
        # predict tracks
        for trk in track_list:
            trk.predict()

        # association, giving priority to primary tracks
        matched_tracks_id, matched_dets_id = associate_main(track_list, cur_dets)

        tracks_ids_to_delete = []
        for i, trk in enumerate(track_list):
            trk.life_count += 1
            trk.tentative = False if trk.life_count >= 5 else True
            if trk.id in matched_tracks_id:
                det_index = matched_dets_id[matched_tracks_id.index(trk.id)]
                det = pd.Series(cur_dets.loc[det_index, :])
                update_matched_track(trk, det)
            else:
                trk.match = False
                trk.unmatch_count += 1
                if (trk.tentative and trk.unmatch_count >= 1) or trk.unmatch_count >= 5:
                    tracks_ids_to_delete.append(trk.id)

            # delete tracks exiting FoV
            is_exiting_fov = (trk.bb[1] <= 0 and trk.is_moving_left()) or (trk.bb[3] >= 1200 and not trk.is_moving_left())
            if is_exiting_fov:
                tracks_ids_to_delete.append(trk.id)

        # new tracks from unassociated detections
        for det_id in cur_dets.index:
            if det_id not in matched_dets_id:
                det = pd.Series(cur_dets.loc[det_id, :])
                new_trk = create_new_track(track_list,det, cycle_time)
                track_list.append(new_trk)

        # delete tracks
        if len(tracks_ids_to_delete) > 0:
            for i, trk in enumerate(track_list):
                if trk.id in tracks_ids_to_delete:
                    del track_list[i]

    return track_list


def update_matched_track(trk, det):
    trk.match = True
    trk.unmatch_count = 0
    meas = np.array([det['center_x'].item(), det['center_y'].item(), det['width'].item(), det['height'].item()])
    trk.update(meas)


def format_tracks_for_eval(tracks_df, dataset='KITTI'):
    config = get_config(dataset)
    
    tracks_df.loc[:, 'truncated'] = -1
    tracks_df.loc[:, 'occluded'] = -1
    tracks_df.loc[:, 'alpha'] = 0.0
    tracks_df.loc[:, 'height_m'] = 2
    tracks_df.loc[:, 'width_m'] = 3
    tracks_df.loc[:, 'length_m'] = 5
    tracks_df.loc[:, 'x'] = -1
    tracks_df.loc[:, 'y'] = -1
    tracks_df.loc[:, 'z'] = -1
    tracks_df.loc[:, 'rotation_y'] = -1
    tracks_df.loc[:, 'score'] = 0.5 # TODO: fill later

    tracks_df = tracks_df[config['track_cols']]

    return tracks_df

