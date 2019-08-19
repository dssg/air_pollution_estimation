import matplotlib.pyplot as plt
import pandas as pd

from traffic_analysis.d06_visualisation.plotting_helpers import safe_save_plt


def plot_map_over_time(frame_level_map: pd.DataFrame,
                       show_plot: bool = True,
                       save_path: str = None):
    """Plots mean average precision for a chunk of videos as a line plot, with a 
    different line for each vehicle type. 

    Args: 
    frame_level_map: output of ChunkEvaluator.evaluate_frame_level()
    show_plot: If true, will display the plot 
    save_path: if specified, will save to this location (specify full path with 
    desired filename) 
    """
    frame_level_map = frame_level_map.sort_values(
        by='video_upload_datetime', ascending=True)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index

    for vehicle_type, vehicle_group_df in frame_level_map.groupby("vehicle_type"):
        plt.plot(vehicle_group_df["video_upload_datetime"],
                 vehicle_group_df["mean_avg_precision"],
                 label=vehicle_type,
                 marker='o')

    # aesthetics
    ax.set_facecolor('whitesmoke')
    for side in ax.spines:
        ax.spines[side].set_visible(False)

    plt.legend(loc='lower right')
    plt.title("Mean Average Precision Over Time")
    plt.xlabel("Video Upload DateTime")
    plt.ylabel("Mean Average Precision")
    plt.subplots_adjust(bottom=0.25)

    if save_path is not None:
        safe_save_plt(plt=plt, save_path=save_path)
    if show_plot:
        plt.show()

    plt.close()
