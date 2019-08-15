import os
import re

def get_project_directory() -> list:
    """ Returns project directory
        Returns:
            project_dir (str): string containing the directory of the project
    """
    project_dir = re.split(r"\\|/",
                           os.path.dirname(os.path.realpath(__file__))) + ['..', '..', '..']

    return project_dir
