import os
from abc import ABC
import numpy as np
import pandas as pd
import librosa

base_folder_path = '/mnt/telecom1/emotional_speech_datasets'


class EmotionalSpeechDataset(ABC):
    """
    Base class for emotional speech datasets
    Implements basic methods to interact/filter such data
    based on emotions ans speakers. More to be added as and when required.
    """
    def __init__(self, data_dir_name):
        self.data_path = os.path.join(base_folder_path, data_dir_name)
        self.metadata_csv_path = os.path.join(self.data_path,
                                              data_dir_name.replace(" ", "") + '.csv')
        self.speakers = None
        self.emotions = None
        self.metadata = None

    def create_metadata_csv(self):
        pass

    def load_metadata_csv(self):
        if not os.path.exists(self.metadata_csv_path):
            self.create_metadata_csv()
        self.metadata = pd.read_csv(self.metadata_csv_path)
        return

    def get_speaker_emotion(self, speaker_list, emotion_list):
        """
        Filter pandas data-frame based on provided speaker and emotion
        lists
        Parameters
        ----------
        speaker_list: list
            list of speakers to filter
        emotion_list: list
            list of emotions to filter

        Returns
        -------
            pandas data frame (filtered)
        """
        filtered_metadata = self.metadata[(self.metadata['speaker'].
                                           isin(speaker_list)) &
                                          (self.metadata['emotion'].
                                           isin(emotion_list))]
        return filtered_metadata


class EmovDB(EmotionalSpeechDataset):

    def __init__(self, data_dir_name):
        super().__init__(data_dir_name)

    def create_metadata_csv(self):
        """
        Writes a metadata csv
        Code adapted from dataset github repo script align_db.py

        Returns
        -------
        An output metadata file stored at self.metadata_csv_path
        """

        data = []
        # get a list of speakers, in emovdb this can be obtained by checking
        # subdirectory names
        self.speakers = next(os.walk(self.data_path))[1]
        # print(self.speakers)

        for spk in self.speakers:
            # maybe all emotions do not exist for all speaker
            emo_cat = next(os.walk(os.path.join(self.data_path, spk)))[1]
            for emo in emo_cat:
                for file in os.listdir(os.path.join(self.data_path, spk, emo)):
                    # print(file)
                    fpath = os.path.join(self.data_path, spk, emo, file)

                    # check if it is a wav file
                    if file[-4:] == '.wav':
                        e = {'id': file[:-4],
                             'speaker': spk,
                             'emotion': emo,
                             'file_path': fpath}
                        data.append(e)
        data = pd.DataFrame.from_records(data)
        self.emotions = data.emotion.unique().tolist()
        data.to_csv(self.metadata_csv_path, index=False)
        print('data written to csv file')
        return


class Ravdess(EmotionalSpeechDataset):

    def __init__(self, data_dir_name):
        super().__init__(data_dir_name)
        self.emotion_dict = {'01': 'neutral',
                             '02': 'calm',
                             '03': 'happy',
                             '04': 'sad',
                             '05': 'angry',
                             '06': 'fearful',
                             '07': 'disgust',
                             '08':  'surprised'}
        self.intensity_dict = {'01': 'normal',
                               '02': 'strong'}
        self.sentence_dict = {'01': 'Kids are talking by the door',
                              '02': 'Dogs are sitting by the door'}
        self.repetition_dict = {'01' : '1', '02': '2'}

    def create_metadata_csv(self):
        data = []
        self.speakers = next(os.walk(self.data_path))[1]
        # print(self.speakers)
        for spk in self.speakers:
            for file in os.listdir(os.path.join(self.data_path, spk)):
                fpath = os.path.join(self.data_path, spk, file)
                if file[-4:] == '.wav':
                    fid = file[:-4]
                    fid_split = fid.split('-')
                    assert(fid_split[0] == '03')
                    assert(fid_split[1] == '01')

                    e = {'id': fid,
                         'speaker': spk,
                         'emotion': self.emotion_dict[fid_split[2]],
                         'intensity': self.intensity_dict[fid_split[3]],
                         'sentence': self.sentence_dict[fid_split[4]],
                         'repetition': self.repetition_dict[fid_split[5]],
                         'gender': ['male' if int(fid_split[6]) % 2 else 'female'],
                         'file_path': fpath}
                    data.append(e)
        data = pd.DataFrame.from_records(data)
        self.emotions = data.emotion.unique().tolist()
        data.to_csv(self.metadata_csv_path, index=False)
        print('data written to csv file')
        return


def compute_mel_spectrogram(wav_paths_list, n_fft=1024, hop_length=256, n_mels=80, **kwargs):
    """
    Computing mel spec for input wav files. Current params chosen as per
    compatibility with Waveglow

    Parameters
    ----------
    wav_paths_list: list
        list of wav files to process
    n_fft: int
    hop_length: int
    n_mels : int
    kwargs: dict
        other input arguments for mel spec computation function

    Returns
    -------

    """
    n_features = []
    for wav_file in wav_paths_list:
        x, sr = librosa.load(wav_file)
        x_mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, power=1.0,
                                                    n_fft=n_fft, hop_length=hop_length,
                                                    n_mels=n_mels, **kwargs)
        assert x_mel_spec.shape[0] == n_mels
        x_out = dynamic_range_compression(x_mel_spec)
        n_features.append(x_out)
    return n_features


def dynamic_range_compression(x, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None))
