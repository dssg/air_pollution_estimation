import urllib
import json
import csv
import os

website = 'https://api.tfl.gov.uk/Place/Type/JamCam'
res = urllib.request.urlopen(website)
data = json.loads(res.read())

fields_to_remove = ['$type', 'url',
                    'placeType',
                    'children',
                    'childrenUrls']

processed_data = []

for cam in data:
    processed_cam = {}
    for val in cam:
        if (val == 'additionalProperties'):
            for prop in cam[val]:
                if(prop['key'] != 'available'):
                    processed_cam[prop['key']] = prop['value']
        elif(val not in fields_to_remove):
            processed_cam[val] = cam[val]

    processed_data.append(processed_cam)

save_dir = os.path.join(os.getcwd(),'..', '..', 'data/02_processed/JamCam/JamCamDetails.csv')
keys = processed_data[0].keys()
with open(save_dir, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(processed_data)