import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TRACKER_NAME = 'customSORT'
# TRACKER_NAME = 'customSORT_pa_lowr_exfov'
TRACKER_NAME = 'customSORT_v1_newDeletion'
RESULTS_FOLDER = f'data/tracks'
CLASS = "pedestrian"   # car, pedestrian
COLS = ['HOTA', 'MOTA', 'IDSW', 'IDTP', 'IDFN', 'IDFP', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'Dets', 'GT_Dets', 'IDs', 'GT_IDs']

def print_summary_metrics():
    car_res = pd.read_csv(f'{RESULTS_FOLDER}/car_summary.txt', sep=" ")
    ped_res = pd.read_csv(f'{RESULTS_FOLDER}/pedestrian_summary.txt', sep=" ")
    all_classes = pd.concat([car_res, ped_res])
    all_classes.index = ['Car', 'Pedestrian']
    print(all_classes[COLS])

def print_detailed_metrics():
    cols_detailed = ['seq', 'HOTA___AUC', 'IDSW', 'Frag', 'GT_IDs']
    car_res = pd.read_csv(f'{RESULTS_FOLDER}/{CLASS}_detailed.csv', sep=",")
    car_res = car_res[cols_detailed].sort_values(by='GT_IDs', ascending=False)
    print(car_res)

def plot_metrics(trk1_name, trk2_name, class_="car"):
    res_1 = pd.read_csv(f'{RESULTS_FOLDER}/{trk1_name}/{class_}_detailed.csv', sep=",")
    res_2 = pd.read_csv(f'{RESULTS_FOLDER}/{trk2_name}/{class_}_detailed.csv', sep=",")
    res_1 = res_1.loc[res_1["seq"]=="COMBINED", :].reset_index()
    res_2 = res_2.loc[res_2["seq"]=="COMBINED", :].reset_index()

    overall_metrics = ["HOTA___AUC", "DetA___AUC", "AssA___AUC", "DetRe___AUC", "AssRe___AUC"]
    absolute_metrics= ["IDSW", "Frag"]
    bar_width = 0.27 

    plot_config = [
        {"metrics": overall_metrics, "as_int": False, "top": 1},
        {"metrics": absolute_metrics, "as_int": True, "top": 900},
    ]

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2)
    for i, pconf in enumerate(plot_config):
        ax = axes[i]
        r1, r2 = _plot_comparison_bars(ax, res_1, res_2, pconf["metrics"], bar_width)
        if i == 0:
            ax.legend((r1[0], r2[0]), (_get_trk_suffix(trk1_name), _get_trk_suffix(trk2_name)))
        ax.set_ylim(top = pconf["top"])
        _autolabel(ax, r1, pconf["as_int"])
        _autolabel(ax, r2, pconf["as_int"])
    plt.show()


_get_trk_suffix = lambda x: "_".join(x.split("_")[1:])


def _autolabel(ax, rects, as_int = False):
    for rect in rects:
        h = rect.get_height()
        h_viz = str(int(h)) if as_int else str(round(h, 2))
        ax.text(rect.get_x()+rect.get_width()/2., 1.01*h, h_viz,
                ha='center', va='bottom')


def _plot_comparison_bars(ax, res_1, res_2, metrics, bar_width):
    bar_ind = np.arange(len(metrics))
    bar_offset = bar_width + 0.1
    rects1 = ax.bar(bar_ind, list(res_1.loc[0, metrics].values), bar_width)
    rects2 = ax.bar(bar_ind + bar_offset, list(res_2.loc[0, metrics].values), bar_width)
    ax.set_xticks(bar_ind + bar_offset/2)
    ax.set_xticklabels(metrics)
    return rects1, rects2


if __name__ == '__main__':
    trk1_name = 'customSORT_v1_newDeletion'
    trk2_name = 'customSORT'
    class_ = "car"   # car, pedestrian
    plot_metrics(trk1_name, trk2_name, class_)