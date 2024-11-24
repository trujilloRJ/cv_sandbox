import pandas as pd

TRACKER_NAME = 'customSORT'
# TRACKER_NAME = 'customSORT_priorityAssociation'
# TRACKER_NAME = 'customSORT_pa_lowr'
# TRACKER_NAME = 'customSORT_pa_lowr_exfov'
RESULTS_FOLDER = f'data/tracks/{TRACKER_NAME}'

def main():
    cols = ['HOTA', 'MOTA', 'IDSW', 'IDTP', 'IDFN', 'IDFP', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'Dets', 'GT_Dets', 'IDs', 'GT_IDs']
    car_res = pd.read_csv(f'{RESULTS_FOLDER}/car_summary.txt', sep=" ")
    ped_res = pd.read_csv(f'{RESULTS_FOLDER}/pedestrian_summary.txt', sep=" ")
    all_classes = pd.concat([car_res, ped_res])
    all_classes.index = ['Car', 'Pedestrian']
    print(all_classes[cols])

if __name__ == '__main__':
    main()