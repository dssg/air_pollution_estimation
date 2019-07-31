import os


def get_project_directory():
    """ Returns project directory
        Returns:
            project_dir (str): string containing the directory of the project
    """
    project_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."
    )

    return project_dir
