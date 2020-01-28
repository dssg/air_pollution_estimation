import os
import re


def get_project_directory() -> list:
    """ Returns project directory
        Returns:
            project_dir (list): path back to top level directory
    """
    project_dir = re.split(r"\\|/",
                           os.path.dirname(os.path.realpath(__file__))) + ['..', '..', '..']

    return project_dir
