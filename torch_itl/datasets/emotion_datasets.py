import os
from abc import ABC
import numpy as np
import pandas as pd

#base_folder_path = '/mnt/telecom1/emotional_speech_datasets'
#base_folder_path = '/home/mlpboon/Downloads'
base_folder_path = '/home/mlpboon/post-doc/repositories/external/GANnotation'
#base_folder_path = '/home/mlpboon/post-doc/repositories/torch_itl/datasets'


class EmotionDataset(ABC):
    """
    Base class for audio/visual emotion datasets
    Implements basic methods to interact/filter such data
    based on emotions and speakers/subjects.
    """
    def __init__(self, data_dir_name, metadata_csv_path='in_data_dir'):
        """

        Parameters
        ----------
        data_dir_name: str
                       data directory name
        metadata_csv_path: str
                           place to output or find meta data csv, pass 'in_data_dir'
                           to place it inside data_dir
        """
        self.data_dir_name = data_dir_name
        self.data_path = os.path.join(base_folder_path, data_dir_name)
        if metadata_csv_path == 'in_data_dir':
            self.metadata_csv_path = os.path.join(self.data_path,
                                              data_dir_name.split('/')[-1].replace(" ", "") + '.csv')
        else:
            self.metadata_csv_path = metadata_csv_path
        self.speakers = None
        self.emotions = None
        self.metadata = None

    def create_metadata_csv(self):
        pass

    def load_metadata_csv(self):
        if not os.path.exists(self.metadata_csv_path):
            self.create_metadata_csv()
        self.metadata = pd.read_csv(self.metadata_csv_path)
        self.speakers = self.metadata['speaker'].unique().tolist()
        # self.emotions = self.metadata['emotion'].unique().tolist()
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


class KdefData(EmotionDataset):
    def __init__(self, data_dir_name):
        super(KdefData, self).__init__(data_dir_name)
        self.sessions = ['A', 'B']
        self.gender = ['F', 'M']
        self.num_actors_per_gender = 35
        self.emotions = ['AF', 'AN', 'DI', 'HA', 'SA', 'SU']
        self.profile = ['S', 'FL', 'HL', 'HR', 'FR']
        self.extension = '.JPG'

    def create_metadata_csv(self):
        data = []
        for i, sess in enumerate(self.sessions):
            for j, gen in enumerate(self.gender):
                for k in range(self.num_actors_per_gender):
                    folder_name = sess + gen + str(k + 1).zfill(2)
                    folder_path = os.path.join(self.data_path, folder_name)

                    for e, emo in enumerate(['NE'] + self.emotions):
                        for pr in self.profile:
                            emotion_image_path = os.path.join(folder_path, folder_name + emo + pr +
                                                              self.extension)
                            if os.path.exists(emotion_image_path):
                                dr = {'id': folder_name + emo + pr,
                                      'speaker': gen + str(k + 1).zfill(2),
                                      'emotion': emo,
                                      'session': sess,
                                      'gender': 'male' if gen == 'M' else 'female',
                                      'profile': pr,
                                      'file_path': emotion_image_path}
                                data.append(dr)
                            else:
                                print(emotion_image_path)
        data = pd.DataFrame.from_records(data)
        self.emotions = data.emotion.unique().tolist()
        data.to_csv(self.metadata_csv_path, index=False)
        print('data written to csv file')


class RafdData(EmotionDataset):
    def __init__(self, data_dir_name):
        super(RafdData, self).__init__(data_dir_name)
        self.group = ['Caucasian', 'Moroccan']
        self.gender = ['female', 'male']
        self.age = ['Kid', 'Adult']
        self.emotions = ['happy', 'angry', 'sad', 'contemptuous', 'disgusted', 'neutral', 'fearful', 'surprised']
        self.profile = ['000', '045', '090', '135', '180']
        self.gaze = ['left', 'frontal', 'right']
        self.extension = '.jpg'

    def create_metadata_csv(self):
        data = []

        for pic in sorted(os.listdir(self.data_path)):
            if pic.split('.')[-1].lower() == self.extension[1:]:
                fname_part = pic.split('.')[0].split('_')
                assert fname_part[0][4:] in self.profile
                assert (fname_part[2] in self.group or fname_part[2] == self.age[0])
                assert fname_part[3] in self.gender
                assert fname_part[4] in self.emotions
                assert fname_part[5] in self.gaze

                dr = {'id': pic.split('.')[0],
                      'speaker': fname_part[1],
                      'profile': fname_part[0][4:],
                      'group': 'Caucasian' if fname_part[2] == self.age[0] else fname_part[2],
                      'age': fname_part[2] if fname_part[2] == self.age[0] else self.age[1],
                      'gender': fname_part[3],
                      'emotion': fname_part[4],
                      'gaze': fname_part[5],
                      'file_path': os.path.join('.', self.data_dir_name, pic)
                     }
                data.append(dr)
        data = pd.DataFrame.from_records(data)
        data.to_csv(self.metadata_csv_path, index=False)
        print('data written to csv file')
        return


if __name__ == "__main__":
    rafd = RafdData('RafD')
    rafd.create_metadata_csv()
