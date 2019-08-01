import matplotlib.pyplot as plt
import pandas as pd


def plot_video_stats_diff_distribution(video_level_diff_df: pd.DataFrame,
                                       show_plot: bool = True,
                                       save_path: str = None):
	"""For each statistic type (counts, starts, stops, etc.) plots the
	distribution of difference from ground truth for each vehicle type

	Args:
		video_level_diff_df: the output of ChunkEvaluator.evaluate_video_level()
		show_plot: If true, will display the plot
		save_path: if specified, will save to this location (specify full path with
		desired filename)
	"""
    plt.style.use('seaborn-deep')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(25,20))
    for i,stat_type in enumerate(params["video_level_stats"]):
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
		                           metrics = {'mean_diff': "", 'mse':""},
		                           show_plots = True): 
	"""For each error metric specified, will plot a multi-bar bar chart with bars 
	for each vehicle type and stats type.

	Args: 
		video_level_stats_df: output of ChunkEvaluator.aggregate_video_stats_all_vehicle_types()
		metrics: dictionary where the metric type is the key and the value is the desired save path
		show_plots: If true, will display the plots 
	"""
    n_videos = video_level_stats_df['n_videos'].iloc[0]
    
    if 'mean_diff' in metrics: 
        df = video_level_stats_df[video_level_stats_df["statistic"] != "mse"]
        mean_df = df[df['statistic'] == 'mean_diff'][['counts','stops','starts','parked','vehicle_type']].set_index('vehicle_type')
        sd_df = df[df['statistic'] == 'sd'][['counts','stops','starts','parked','vehicle_type']].set_index('vehicle_type').values.T

        ax = mean_df.plot(kind='bar',
                            yerr=sd_df,
                            grid=False,
                            figsize=(10,8),
                            position=0.45,
                            colormap = 'Paired',
                            error_kw=dict(ecolor='k',elinewidth=0.5),
                            width=1.0
                            )
        # aesthetics 
        ax.set_facecolor('whitesmoke')
        for side in ax.spines:
            ax.spines[side].set_visible(False)
        plt.xticks(rotation=45)

        # titles 
        plt.title(f"Video level performance on {n_videos} videos")
        plt.xlabel("Vehicle Type", labelpad=20)
        plt.ylabel("Mean Difference")

        if show_plots: 
            plt.show()
        if metrics["mean_diff"] is not None: 
            plt.savefig(metrics["mean_diff"])
        plt.close()
        
    if 'mse' in metrics: 
        df = video_level_stats_df[video_level_stats_df["statistic"] == "mse"]
        mse_df = df[['counts','stops','starts','parked','vehicle_type']].set_index('vehicle_type')
        ax = mse_df.plot(kind='bar',
                            grid=False,
                            figsize=(10,8),
                            position=0.45,
                            colormap = 'Paired',
                            width=1.0
                            )
        # aesthetics
        plt.xticks(rotation=45)
        ax.set_facecolor('whitesmoke')
        for side in ax.spines:
            ax.spines[side].set_visible(False)

        # titling
        plt.title(f"Video level performance on {n_videos} videos")
        plt.xlabel("Vehicle Type", labelpad=20)
        plt.ylabel("MSE")

        if show_plots: 
            plt.show()
        if metrics["mse"] is not None: 
            plt.savefig(metrics["mse"])
        plt.close()


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
