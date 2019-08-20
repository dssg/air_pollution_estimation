import yaml
import os
import re
from traffic_analysis.d00_utils.get_project_directory import get_project_directory


project_dir = get_project_directory()


def collapse_dict_hierarchy(nested_dict: dict):
    collapsed_dict = {}
    for name, subdict in nested_dict.items():
        collapsed_dict = {**collapsed_dict, **subdict}
    return collapsed_dict


def load_parameters():
    filepath = os.sep.join(project_dir + 'conf/base/parameters.yml'.split('/'))
    with open(filepath) as f:
        params = yaml.safe_load(f)
    return collapse_dict_hierarchy(params)


def load_app_parameters():
    filepath = os.sep.join(
        project_dir + 'conf/base/app_parameters.yml'.split('/'))

    with open(filepath) as f:
        params = yaml.safe_load(f)
    return {**params['visualization']}


def load_credentials():

    filepath = os.sep.join(
        project_dir + 'conf/local/credentials.yml'.split('/'))

    with open(filepath) as f:
        creds = yaml.safe_load(f)

    return creds


def load_paths():
    filepath = os.sep.join(project_dir + 'conf/base/paths.yml'.split('/'))

    with open(filepath) as f:
        paths = yaml.safe_load(f)
        s3_paths = paths['s3_paths']
        local_paths = paths['local_paths']
        db_paths = paths['db_paths']

    for key, val in local_paths.items():
        local_paths[key] = os.sep.join(project_dir + re.split(r"\\|/", val))
    return {**s3_paths, **local_paths, **db_paths}
