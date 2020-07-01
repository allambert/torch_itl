import numpy as np
import torch
import pandas as pd

from .synthetic_quantile import toy_data_quantile
from .emotional_speech_datasets import Ravdess
from .emotional_speech_dataloader import generate_training_samples

def import_data_otoliths():

    filename_prefix = './datasets/otoliths/train/Train_fc_'
    filename_suffix = '.npy'
    n = 3780
    X_train = np.zeros((n, 64 * 64))
    for i in range(n):
        X_train[i] = np.load(filename_prefix + str(i) + filename_suffix)
    training_info = pd.read_csv('./datasets/otoliths/train/training_info.csv')
    Y_train = training_info['Ground truth'].values
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()

    filename_prefix = './datasets/otoliths/test/Test_fc_'
    filename_suffix = '.npy'
    n = 165
    X_test = np.zeros((n, 64 * 64))
    for i in range(n):
        X_test[i] = np.load(filename_prefix + str(i) + filename_suffix)
    testing_info = pd.read_csv('./datasets/otoliths/test/Testing_info.csv')
    Y_test = testing_info['Ground truth'].values
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()
    return X_train, Y_train, X_test, Y_test

def import_data_toy_quantile(*args):
    x_train, y_train, probs = toy_data_quantile(*args)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    probs = torch.Tensor(probs)
    return x_train, y_train, probs


def import_speech_synth_ravdess():
    db_rds = Ravdess('RAVDESS')
    db_rds.load_metadata_csv()
    print(db_rds.speakers)
    set_speakers = ['Actor_01']
    set_emotions = ['neutral', 'sad', 'happy', 'calm']
    set_sentence = db_rds.sentence_dict['01']
    set_repetition = int(db_rds.repetition_dict['01'])
    set_intensity = db_rds.intensity_dict['01']
    db_rds_filtered = db_rds.get_speaker_emotion(set_speakers, set_emotions)
    train_db_rds = db_rds_filtered[(db_rds_filtered['sentence'] == set_sentence) &
                                   (db_rds_filtered['repetition'] == set_repetition) &
                                   (db_rds_filtered['intensity'] == set_intensity)]
    inp_list = [[train_db_rds[train_db_rds['emotion'] == 'neutral']['file_path'].tolist()[0],
                 train_db_rds[train_db_rds['emotion'] != 'neutral']['file_path'].tolist()]]
    """
    inp_list = [['./RAVDESS/Actor_01/03-01-01-01-01-01-01.wav',
                 ['./RAVDESS/Actor_01/03-01-02-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-03-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-04-01-01-01-01.wav']]]
    """
    training_samples = generate_training_samples(inp_list, context=0)
    # x_train = torch.from_numpy(np.expand_dims(training_samples[0][0], axis=1)).float()
    x_train = torch.from_numpy(np.squeeze(training_samples[0][0])).float()
    y_train = torch.from_numpy(training_samples[0][1]).float()
    return x_train, y_train
