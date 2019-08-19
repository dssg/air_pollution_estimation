import matplotlib.pyplot as plt
import pandas as pd

from traffic_analysis.d06_visualisation.plotting_helpers import safe_save_plt


def plot_video_level_performance(video_level_performance: pd.DataFrame,
                                 metrics={'bias': None,
                                          'rmse': None,
                                          'mae': None},
                                 show_plots=True):
    """For each error metric specified, will plot a multi-bar bar chart with bars 
    for each vehicle type and stats type.

    Args: 
        video_level_performance: output of ChunkEvaluator.evaluate_video_level()
        metrics: dictionary where the metric type is the key and the value is the desired save path
        show_plots: If true, will display the plots 
    """
    assert set(metrics.keys()).issubset(set(['bias', 'rmse', 'mae'])), \
        "Only the following metrics types are supported: bias, rmse, mae"
    n_videos = video_level_performance['n_videos'].iloc[0]
    video_level_performance.columns = video_level_performance.columns.str.lower()

    def style_show_save_plot(metric_type):
        ax.set_facecolor('whitesmoke')

        for side in ax.spines:
            ax.spines[side].set_visible(False)

        plt.title(f"Video level performance on {n_videos} videos")
        plt.xticks(rotation=45)
        plt.xlabel("Vehicle Type", labelpad=20)
        plt.ylabel(metric_type)
        plt.ylim(ymax=25, ymin=0)
        plt.subplots_adjust(bottom=0.25)

        if metrics[metric_type] is not None:
            safe_save_plt(plt=plt, save_path=metrics[metric_type])
        if show_plots:
            plt.show()
        plt.close()

    if 'bias' in metrics:
        bias_df = (video_level_performance[['stat', 'vehicle_type', 'bias']]
                   .pivot(index='vehicle_type', columns='stat', values='bias'))
        sd_df = (video_level_performance[['stat', 'vehicle_type', 'sd']]
                 .pivot(index='vehicle_type', columns='stat', values='sd')
                 .values.T)

        ax = bias_df.plot(kind='bar',
                          yerr=sd_df,
                          grid=False,
                          figsize=(12, 10),
                          position=0.45,
                          colormap='Paired',
                          error_kw=dict(ecolor='k', elinewidth=0.5),
                          width=1.0
                          )
        style_show_save_plot(metric_type='bias')

    if 'mae' in metrics:
        mae_df = (video_level_performance[['stat', 'vehicle_type', 'mae']]
                  .pivot(index='vehicle_type', columns='stat', values='mae'))
        sd_df = (video_level_performance[['stat', 'vehicle_type', 'sd']]
                 .pivot(index='vehicle_type', columns='stat', values='sd')
                 .values.T)
        ax = mae_df.plot(kind='bar',
                         yerr=sd_df,
                         grid=False,
                         figsize=(12, 10),
                         position=0.45,
                         colormap='Paired',
                         error_kw=dict(ecolor='k', elinewidth=0.5),
                         width=1.0
                         )

        style_show_save_plot(metric_type='mae')

    if 'rmse' in metrics:
        rmse_df = (video_level_performance[['stat', 'vehicle_type', 'rmse']]
                   .pivot(index='vehicle_type', columns='stat', values='rmse'))

        ax = rmse_df.plot(kind='bar',
                          grid=False,
                          figsize=(12, 10),
                          position=0.45,
                          colormap='Paired',
                          width=1.0
                          )
        style_show_save_plot(metric_type='rmse')
