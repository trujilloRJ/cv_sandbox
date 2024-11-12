import pandas as pd 
import numpy as np
from track_maintenance import create_new_track
from association import associate_tracks_dets


def do_track_cyclic(track_list, cur_dets, cycle, cycle_time):
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

        # associate
        matched_tracks_id, matched_dets_id = associate_tracks_dets(track_list, cur_dets)

        # update associated tracks and track maintenance
        tracks_ids_to_delete = []
        for i, trk in enumerate(track_list):
            trk.life_count += 1
            trk.tentative = False if trk.life_count >= 5 else True
            if i in matched_tracks_id:
                trk.match = True
                trk.unmatch_count = 0

                det_index = matched_dets_id[matched_tracks_id.index(i)]
                det = pd.Series(cur_dets.loc[det_index, :])
                meas = np.array([det['center_x'].item(), det['center_y'].item(), det['width'].item(), det['height'].item()])
                trk.update(meas)
            else:
                trk.match = False
                trk.unmatch_count += 1
                if (trk.tentative and trk.unmatch_count >= 1) or trk.unmatch_count >= 5:
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


def format_tracks_for_eval(tracks_df):
    cols = ['frame', 'id', 'type', 'truncated', 'occluded', 'alpha', 
                  'left', 'top', 'right', 'bottom', 
                  'height_m', 'width_m', 'length_m',
                  'x', 'y', 'z', 'rotation_y', 'score']
    
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

    tracks_df = tracks_df[cols]

    return tracks_df

