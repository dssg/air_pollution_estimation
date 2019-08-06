import matplotlib.pyplot as plt
import pandas as pd


def plot_video_stats_diff_distribution(video_level_diff_df: pd.DataFrame,
                                       video_stat_types: list, 
                                       show_plot: bool = True,
                                       save_path: str = None):
    """For each statistic type (counts, starts, stops, etc.) plots the
    distribution of difference from ground truth for each vehicle type

    Args:
        video_level_diff_df: the output of ChunkEvaluator.evaluate_video_level()
        video_stat_types: list of video level stats computed (pass from params)
        show_plot: If true, will display the plot
        save_path: if specified, will save to this location (specify full path with
        desired filename)
    """
    plt.style.use('seaborn-deep')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(25,20))
    
    for i,stat_type in enumerate(video_stat_types):
        stat_list = []
        vehicle_types=[]
        for vehicle_type, vehicle_group_df in video_level_diff_df.groupby("vehicle_type"):
            stat_list.append(vehicle_group_df[f"y_pred-y_{stat_type}"])
            vehicle_types.append(vehicle_type)

        # plot distribution of stat_type on one sub-plot
        ax = eval(f"ax{i+1}")
        ax.set_title(stat_type)
        ax.hist(x=stat_list, 
                label=vehicle_types,
                alpha=0.8)
        ax.legend(loc="upper right")

    fig.suptitle("Distribution of Differences for Traffic Statistics", size=24)
    fig.text(0, 0.5,"Frequency", ha='center', va='center', rotation='vertical', size=24) #y label
    fig.text(0.5, 0, "Model Prediction", ha='center', va='center', size=24) # x label

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, 
                        left = 0.05,
                        right= 0.90,
                        bottom = 0.05
                       )
    if show_plot: 
        plt.show()
    if save_path is not None: 
        plt.savefig(path)
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
        if show_plots: 
            plt.show()
        if metrics[metric_type] is not None: 
            plt.savefig(metrics[metric_type])
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
                            figsize=(10,8),
                            position=0.45,
                            colormap = 'Paired',
                            error_kw=dict(ecolor='k',elinewidth=0.5),
                            width=1.0
                            )
        style_show_save_plot(metric_type = 'bias')
        
    if 'mae' in metrics: 
        mae_df = (video_level_stats_df[['stat','vehicle_type', 'MAE']]
                    .pivot(index='vehicle_type', columns='stat', values='MAE'))
        sd_df = (video_level_stats_df[['stat','vehicle_type', 'sd']]
                    .pivot(index='vehicle_type', columns='stat', values='sd')
                    .values.T)
        ax = mae_df.plot(kind='bar',
                            yerr=sd_df,
                            grid=False,
                            figsize=(10,8),
                            position=0.45,
                            colormap = 'Paired',
                            error_kw=dict(ecolor='k',elinewidth=0.5),
                            width=1.0
                            )

        style_show_save_plot(metric_type = 'mae')
        
    if 'rmse' in metrics: 
        rmse_df = (video_level_stats_df[['stat','vehicle_type', 'RMSE']]
                    .pivot(index='vehicle_type', columns='stat', values='RMSE'))

        ax = rmse_df.plot(kind='bar',
                            grid=False,
                            figsize=(10,8),
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

    fig = plt.figure(figsize = (10,7))
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
    
    if show_plot: 
        plt.show()
    if save_path is not None: 
        plt.savefig(path)
        
    plt.close()
