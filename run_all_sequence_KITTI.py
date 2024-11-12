import glob
import pandas as pd
import numpy as np
from track_main import do_track_cyclic, format_tracks_for_eval
from configuration import get_config


DET_FOLDER = f"data/nms/"
DATASET = 'KITTI'
SAVE_PATH = f'data/tracks/'


if __name__ == "__main__":
    config = get_config(DATASET)
    det_list = [f for f in glob.glob(f"{DET_FOLDER}*.csv")]
    fps = config['FPS']
    cycle_time = 1/fps

    for k, det_seq in enumerate(det_list):
        print(f"{k+1}/{len(det_list)}")
        seq_name = det_seq[-8:-4]

        # load detections
        dets = pd.read_csv(det_seq, sep=' ')
        frames = np.unique(dets['frame'])
        global_track_list = []
        track_list = []
        
        for cyc, frame_index in enumerate(frames):
            # filter detections
            cur_dets = dets.loc[dets['frame'] == frame_index, :]

            # tracking cycle here
            track_list = do_track_cyclic(track_list, cur_dets, cyc, cycle_time)

            for trk in track_list:
                trk_dict = trk.to_dict()
                trk_dict.update({'frame': cyc})
                global_track_list.append(trk_dict)

        # save tracks
        tracks_df = pd.DataFrame.from_records(global_track_list)
        tracks_df = format_tracks_for_eval(tracks_df)
        tracks_df.to_csv(f"{SAVE_PATH}{seq_name}.txt", index=None, header=None, sep=' ')