import yaml
import os 
from traffic_analysis.d00_utils.get_project_directory import get_project_directory


project_dir = get_project_directory()


def collapse_dict_hierarchy(nested_dict: dict):
    collapsed_dict = {}
    for name, subdict in nested_dict.items():
        collapsed_dict = {**collapsed_dict, **subdict}
    return collapsed_dict


def load_parameters():
    with open(project_dir + '/conf/base/parameters.yml') as f:
        params = yaml.safe_load(f)
    return collapse_dict_hierarchy(params)


def load_app_parameters():
    with open(project_dir + '/conf/base/app_parameters.yml') as f:
        params = yaml.safe_load(f)
    return {**params['visualization']}


def load_credentials():

    with open(project_dir + '/conf/local/credentials.yml') as f:
        creds = yaml.safe_load(f)

    return creds


def load_paths():
    with open(project_dir + '/conf/base/paths.yml') as f:
        paths = yaml.safe_load(f)
        s3_paths = paths['s3_paths']
        local_paths = paths['local_paths']
        db_paths = paths['db_paths']

    for key, val in local_paths.items():
        val = os.path.normpath(val)
        if key[:4] == "temp":
            local_paths[key] = os.path.join(project_dir, val, str(os.getpid()))  + os.sep
        else: 
            local_paths[key] = os.path.join(project_dir, val) + os.sep

    return {**s3_paths, **local_paths, **db_paths}
