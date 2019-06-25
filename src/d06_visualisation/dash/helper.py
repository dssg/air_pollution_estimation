import json
import os


def get_cams():
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '..', '..', '..', 'data/00_ref/cam_file.json')

    data = json.loads(open(filepath, 'r').read())
    cam_list = [{'label': item['commonName'],  'value': item['id']}
                for item in dict(data).values()]
    return cam_list


if __name__ == "__main__":
    get_cams()
