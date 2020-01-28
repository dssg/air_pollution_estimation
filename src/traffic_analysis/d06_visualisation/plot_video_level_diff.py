from math import ceil

import matplotlib.pyplot as plt
import pandas as pd

from traffic_analysis.d06_visualisation.plotting_helpers import safe_save_plt


def plot_video_stats_diff_distribution(video_level_diff: pd.DataFrame,
                                       video_stat_types: list = ["counts", "stops", "starts"],
                                       show_plot: bool = True,
                                       save_path: str = None):
    """For each statistic type (counts, starts, stops, etc.) plots the
    distribution of difference from ground truth for each vehicle type

    Args:
        video_level_diff: the output of ChunkEvaluator.evaluate_video_level()
        video_stat_types: list of video level stats computed (pass from params)
        show_plot: If true, will display the plot
        save_path: if specified, will save to this location (specify full path with
        desired filename)
    """
    plt.style.use('seaborn-deep')
    num_stats = len(video_stat_types)
    fig, axes = plt.subplots(nrows=ceil(num_stats/2),
                             ncols=2,
                             figsize=(30, 25))
    i = 0
    for row in axes:
        for ax in row:
            if i >= num_stats:
                fig.delaxes(axes.flatten()[i])
                continue
            stat_type = video_stat_types[i]
            stat_list = []
            vehicle_types = []
            for vehicle_type, vehicle_group_df in video_level_diff.groupby("vehicle_type"):
                stat_list.append(vehicle_group_df[f"{stat_type}_diff"])
                vehicle_types.append(vehicle_type)

            # plot distribution of stat_type on one sub-plot
            ax.set_title(stat_type, fontsize=32)
            ax.hist(x=stat_list,
                    label=vehicle_types,
                    alpha=0.8)
            ax.legend(loc="upper right")

            i += 1

    fig.suptitle("Distribution of Differences for Traffic Statistics", size=40)
    fig.text(0, 0.5, "Frequency", ha='center', va='center',
             rotation='vertical', size=32)  # y label
    fig.text(0.5, 0, "Model Prediction", ha='center',
             va='center', size=32)  # x label

    fig.tight_layout()
    fig.subplots_adjust(top=0.90,
                        left=0.07,
                        right=0.85,
                        bottom=0.07
                        )
    if save_path is not None:
        safe_save_plt(fig=fig, save_path=save_path)
    if show_plot:
        plt.show()
    plt.close()
