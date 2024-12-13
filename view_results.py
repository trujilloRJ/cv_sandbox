import pandas as pd

# TRACKER_NAME = 'customSORT'
# TRACKER_NAME = 'customSORT_pa_lowr_exfov'
TRACKER_NAME = 'customSORT_v1_newScore'
RESULTS_FOLDER = f'data/tracks/{TRACKER_NAME}'
COLS = ['HOTA', 'MOTA', 'IDSW', 'IDTP', 'IDFN', 'IDFP', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'Dets', 'GT_Dets', 'IDs', 'GT_IDs']

def print_summary_metrics():
    car_res = pd.read_csv(f'{RESULTS_FOLDER}/car_summary.txt', sep=" ")
    ped_res = pd.read_csv(f'{RESULTS_FOLDER}/pedestrian_summary.txt', sep=" ")
    all_classes = pd.concat([car_res, ped_res])
    all_classes.index = ['Car', 'Pedestrian']
    print(all_classes[COLS])

def print_detailed_metrics():
    cols_detailed = ['seq', 'HOTA___AUC', 'IDSW', 'Frag', 'GT_IDs']
    car_res = pd.read_csv(f'{RESULTS_FOLDER}/car_detailed.csv', sep=",")
    car_res = car_res[cols_detailed].sort_values(by='GT_IDs', ascending=False)
    print(car_res)

def main():
    print_detailed_metrics()

if __name__ == '__main__':
    main()