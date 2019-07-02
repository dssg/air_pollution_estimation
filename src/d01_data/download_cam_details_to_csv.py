import urllib.request
import json
import csv
import os

# TODO: Move to a function and integrate into a pipeline

# Get the json from the TFL API
website = 'https://api.tfl.gov.uk/Place/Type/JamCam'
res = urllib.request.urlopen(website)
data = json.loads(res.read())

# List of fields to be removed from the json
fields_to_remove = ['$type', 'url',
                    'placeType',
                    'children',
                    'childrenUrls']

# Remove the fields and create a list of dictionaries for each camera
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

# Save the list of dictionaries as a csv file in the processed data folder
setup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
save_dir = os.path.join(setup_dir, 'data/02_processed/jamcams/JamCamDetails.csv')
keys = processed_data[0].keys()
with open(save_dir, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(processed_data)
