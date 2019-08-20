from traffic_analysis.d00_utils.load_confs import load_paths, load_credentials, \
    load_parameters, load_training_parameters
from traffic_analysis.d04_modelling.transfer_learning.training_data_loader import TrainingDataLoader, TransferDataset
from traffic_analysis.d04_modelling.transfer_learning.train_tensorflow_model import transfer_learn

paths = load_paths()
creds = load_credentials()
params = load_parameters()
train_params = load_training_parameters()

training_data_loader = TrainingDataLoader(datasets=[TransferDataset.cvat, TransferDataset.detrac],
                                          creds=creds,
                                          paths=paths)

fraction_for_training = 0.8
x_train, y_train, x_test, y_test = training_data_loader.get_train_and_test(fraction_for_training)


saved_text_files_dir = paths['temp_annotation']
with open(saved_text_files_dir + 'train.txt', 'w') as f:
    for item in y_train:
        f.write("%s\n" % item)

with open(saved_text_files_dir + 'test.txt', 'w') as f:
    for item in y_test:
        f.write("%s\n" % item)

transfer_learn(paths=paths,
               params=params,
               train_params=train_params,
               train_file='train.txt',
               test_file='test.txt',
               selected_labels=params['selected_labels'])