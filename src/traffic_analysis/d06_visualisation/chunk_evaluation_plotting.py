import matplotlib.pyplot as plt
import pandas as pd
from math import ceil

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
                             figsize=(30,25))
    i = 0 
    for row in axes: 
        for ax in row: 
            if i >= num_stats:
                continue
            stat_type = video_stat_types[i]
            stat_list = []
            vehicle_types=[]
            for vehicle_type, vehicle_group_df in video_level_diff.groupby("vehicle_type"):
                stat_list.append(vehicle_group_df[f"{stat_type}_diff"])
                vehicle_types.append(vehicle_type)

            # plot distribution of stat_type on one sub-plot
            ax.set_title(stat_type, fontsize = 32)
            ax.hist(x=stat_list, 
                    label=vehicle_types,
                    alpha=0.8)
            ax.legend(loc="upper right")
            
            i += 1

    fig.suptitle("Distribution of Differences for Traffic Statistics", size=40)
    fig.text(0, 0.5,"Frequency", ha='center', va='center', rotation='vertical', size=32) # y label
    fig.text(0.5, 0, "Model Prediction", ha='center', va='center', size=32) # x label

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, 
                        left = 0.07,
                        right= 0.85,
                        bottom = 0.07
                       )
    if save_path is not None: 
        plt.savefig(save_path)
    if show_plot: 
        plt.show()
    plt.close()


def plot_video_level_summary_stats(video_level_stats_df: pd.DataFrame, 
                                   metrics = {'bias':None, 'rmse':None, 'mae': None},
                                   show_plots = True): 
    """For each error metric specified, will plot a multi-bar bar chart with bars 
    for each vehicle type and stats type.

    Args: 
        video_level_stats_df: output of ChunkEvaluator.aggregate_video_stats_all_vehicle_types()
        metrics: dictionary where the metric type is the key and the value is the desired save path
        show_plots: If true, will display the plots 
    """
    assert set(metrics.keys()).issubset(set(['bias', 'rmse', 'mae'])), \
        "Only the following metrics types are supported: bias, rmse, mae"
    n_videos = video_level_stats_df['n_videos'].iloc[0]
    
    def style_show_save_plot(metric_type):
        ax.set_facecolor('whitesmoke')

        for side in ax.spines:
            ax.spines[side].set_visible(False)

        plt.title(f"Video level performance on {n_videos} videos")
        plt.xticks(rotation=45)
        plt.xlabel("Vehicle Type", labelpad=20)
        plt.ylabel(metric_type)
        plt.ylim(ymax = 25, ymin = 0)
        plt.subplots_adjust(bottom=0.25)

        if metrics[metric_type] is not None: 
            plt.savefig(metrics[metric_type])
        if show_plots: 
            plt.show()
        plt.close()
    
    if 'bias' in metrics: 
        bias_df = (video_level_stats_df[['stat','vehicle_type', 'bias']]
                    .pivot(index='vehicle_type', columns='stat', values='bias'))
        sd_df = (video_level_stats_df[['stat','vehicle_type', 'sd']]
                    .pivot(index='vehicle_type', columns='stat', values='sd')
                    .values.T)

        ax = bias_df.plot(kind='bar',
                            yerr=sd_df,
                            grid=False,
                            figsize=(12,10),
                            position=0.45,
                            colormap = 'Paired',
                            error_kw=dict(ecolor='k',elinewidth=0.5),
                            width=1.0
                            )
        style_show_save_plot(metric_type = 'bias')
        
    if 'mae' in metrics: 
        mae_df = (video_level_stats_df[['stat','vehicle_type', 'mae']]
                    .pivot(index='vehicle_type', columns='stat', values='mae'))
        sd_df = (video_level_stats_df[['stat','vehicle_type', 'sd']]
                    .pivot(index='vehicle_type', columns='stat', values='sd')
                    .values.T)
        ax = mae_df.plot(kind='bar',
                            yerr=sd_df,
                            grid=False,
                            figsize=(12,10),
                            position=0.45,
                            colormap = 'Paired',
                            error_kw=dict(ecolor='k',elinewidth=0.5),
                            width=1.0
                            )

        style_show_save_plot(metric_type = 'mae')
        
    if 'rmse' in metrics: 
        rmse_df = (video_level_stats_df[['stat','vehicle_type', 'rmse']]
                    .pivot(index='vehicle_type', columns='stat', values='rmse'))

        ax = rmse_df.plot(kind='bar',
                            grid=False,
                            figsize=(12,10),
                            position=0.45,
                            colormap = 'Paired',
                            width=1.0
                            )
        style_show_save_plot(metric_type='rmse')
    

def plot_mAP_over_time(frame_level_mAP_df: pd.DataFrame, 
                       show_plot: bool = True, 
                       save_path: str = None):
    """Plots mean average precision for a chunk of videos as a line plot, with a 
    different line for each vehicle type. 

    Args: 
    frame_level_mAP_df: output of ChunkEvaluator.evaluate_frame_level()
    show_plot: If true, will display the plot 
    save_path: if specified, will save to this location (specify full path with 
    desired filename) 
    """
    frame_level_mAP_df = frame_level_mAP_df.sort_values(by='video_upload_datetime',ascending=True)

    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index

    for vehicle_type, vehicle_group_df in frame_level_mAP_df.groupby("vehicle_type"):
        plt.plot(vehicle_group_df["video_upload_datetime"],
                 vehicle_group_df["mean_avg_precision"],
                 label = vehicle_type,
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
        plt.savefig(save_path)
    if show_plot: 
        plt.show()
        
    plt.close()
