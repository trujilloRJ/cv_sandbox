from track_main import main_cyclic

SEQUENCE = "0013"
DET_FILE = f"data/nms/{SEQUENCE}.csv"
DATASET = 'KITTI'
SAVE_PATH = f'data/tracks/'

# parameters



if __name__ == "__main__":
    tracks_df = main_cyclic(DET_FILE, DATASET, out_tentative=False)
    tracks_df.to_csv(f"{SAVE_PATH}{SEQUENCE}.txt", index=None, header=None, sep=' ')