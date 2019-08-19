import re
import os


def safe_save_plt(plt, save_path: str): 
    """Checks if save directory exists. If not, creates 
    directory and saves to that location. 
    """
    save_dir = os.sep.join(re.split(r"\\|/", save_path)[:-1])

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_path)
