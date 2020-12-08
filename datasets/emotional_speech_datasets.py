import os
from abc import ABC
import numpy as np
import pandas as pd
import cv2

#base_folder_path = '/mnt/telecom1/emotional_speech_datasets'
#base_folder_path = '/home/mlpboon/Downloads'
base_folder_path = '/home/mlpboon/post-doc/repositories/external/GANnotation'
#base_folder_path = '/home/mlpboon/post-doc/repositories/torch_itl/datasets'

class EmotionalSpeechDataset(ABC):
    """
    Base class for emotional speech datasets
    Implements basic methods to interact/filter such data
    based on emotions ans speakers. More to be added as and when required.
    """
    def __init__(self, data_dir_name, metadata_csv_path='in_data_dir'):
        self.data_dir_name = data_dir_name
        self.data_path = os.path.join(base_folder_path, data_dir_name)
        if metadata_csv_path == 'in_data_dir':
            self.metadata_csv_path = os.path.join(self.data_path,
                                              data_dir_name.split('/')[-1].replace(" ", "") + '.csv')
        else:
            self.metadata_csv_path = os.path.join(base_folder_path,
                                              data_dir_name.split('/')[-1].replace(" ", "") + '.csv')
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


class KdefData(EmotionalSpeechDataset):
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

    def get_kdef_metadata_tmp(self):
        if os.path.exists('./KDEF/kdef_frontal_path_list.txt'):
            return 'File already exists'
        kdef_frontal_path_list = []
        for i, sess in enumerate(self.sessions):
            for j, gen in enumerate(self.gender):
                for k in range(self.num_actors_per_gender):
                    folder_name = sess + gen + str(k + 1).zfill(2)
                    folder_path = os.path.join(self.data_path, folder_name)

                    # neutral image
                    neutral_image_path = os.path.join(folder_path, folder_name + 'NE' + self.profile[0] + self.extension)
                    kdef_frontal_path_list.append(neutral_image_path)
                    for e, emo in enumerate(self.emotions):
                        emotion_image_path = os.path.join(folder_path, folder_name + emo + self.profile[0] + self.extension)
                        kdef_frontal_path_list.append(emotion_image_path)

        with open('./datasets/KDEF/kdef_frontal_path_list.txt', 'w') as f:
            for fpath in kdef_frontal_path_list:
                f.write('{}\n'.format(fpath))

    def landmark_training_split(self):
        outer_eye_alignment_idx = [36, 45]
        x_train = []
        y_train = []
        y_tmp = []
        landmarks_path = './datasets/KDEF/KDEF_LANDMARKS'
        for i, sess in enumerate(self.sessions):
            for j, gen in enumerate(self.gender):
                for k in range(self.num_actors_per_gender):
                    file_id = sess + gen + str(k + 1).zfill(2)

                    # neutral image
                    neu_lnd_path = os.path.join(landmarks_path, file_id + 'NE' + self.profile[0] + '.txt')
                    with open(neu_lnd_path, 'r') as file:
                        tmp_ne_feat = np.array([line.split() for line in file], dtype=np.float32)
                    x_train.append(tmp_ne_feat.flatten())
                    trans_to = tmp_ne_feat[outer_eye_alignment_idx]
                    for e, emo in enumerate(self.emotions):
                        emo_lnd_path = os.path.join(landmarks_path, file_id + emo + self.profile[0] + '.txt')
                        with open(emo_lnd_path, 'r') as file:
                            tmp_emo_feat1 = np.array([line.split() for line in file], dtype=np.float32)
                        trans_from = tmp_emo_feat1[outer_eye_alignment_idx]
                        tmp_emo_feat = self.transform_NDPoints(trans_from, trans_to, tmp_emo_feat1)
                        y_tmp.append(tmp_emo_feat.flatten())
                    y_train.append(y_tmp)
                    y_tmp = []
        np.save('./datasets/KDEF/input_landmarks_train.npy', np.array(x_train))
        np.save('./datasets/KDEF/output_landmarks_train.npy', np.array(y_train))

    def landmark_training_split_align2(self):
        outer_eye_alignment_idx = [36, 45]
        x_train = []
        y_train = []
        y_tmp = []
        landmarks_path = './datasets/KDEF/KDEF_LANDMARKS'
        for i, sess in enumerate(self.sessions):
            for j, gen in enumerate(self.gender):
                for k in range(self.num_actors_per_gender):
                    file_id = sess + gen + str(k + 1).zfill(2)

                    # neutral image
                    neu_lnd_path = os.path.join(landmarks_path, file_id + 'NE' + self.profile[0] + '.txt')
                    with open(neu_lnd_path, 'r') as file:
                        tmp_ne_feat = np.array([line.split() for line in file], dtype=np.float32)
                    if (i == 0) and (j == 0) and (k == 0):
                        ref_x, ref_y = tmp_ne_feat[outer_eye_alignment_idx[0], 0], \
                                       tmp_ne_feat[outer_eye_alignment_idx[0], 1]
                    change_x = ref_x - tmp_ne_feat[outer_eye_alignment_idx[0], 0]
                    change_y = ref_y - tmp_ne_feat[outer_eye_alignment_idx[0], 1]
                    tmp_ne_feat[:, 0] = tmp_ne_feat[:, 0] + change_x
                    tmp_ne_feat[:, 1] = tmp_ne_feat[:, 1] + change_y

                    x_train.append(tmp_ne_feat.flatten())
                    trans_to = tmp_ne_feat[outer_eye_alignment_idx]
                    for e, emo in enumerate(self.emotions):
                        emo_lnd_path = os.path.join(landmarks_path, file_id + emo + self.profile[0] + '.txt')
                        with open(emo_lnd_path, 'r') as file:
                            tmp_emo_feat1 = np.array([line.split() for line in file], dtype=np.float32)
                        trans_from = tmp_emo_feat1[outer_eye_alignment_idx]
                        tmp_emo_feat = self.transform_NDPoints(trans_from, trans_to, tmp_emo_feat1)
                        y_tmp.append(tmp_emo_feat.flatten())
                    y_train.append(y_tmp)
                    y_tmp = []
        np.save('./datasets/KDEF/input_landmarks_align2.npy', np.array(x_train))
        np.save('./datasets/KDEF/output_landmarks_align2.npy', np.array(y_train))

    # IN PROGRESS / NOT USABLE YET
    def landmark_relative_training_split(self):
        x_train = []
        y_train = []
        y_tmp = []
        landmarks_path = './datasets/KDEF/KDEF_LANDMARKS'
        for i, sess in enumerate(self.sessions):
            for j, gen in enumerate(self.gender):
                for k in range(self.num_actors_per_gender):
                    file_id = sess + gen + str(k + 1).zfill(2)

                    # neutral image
                    neu_lnd_path = os.path.join(landmarks_path, file_id + 'NE' + self.profile[0] + '.txt')
                    with open(neu_lnd_path, 'r') as file:
                        tmp_ne_feat = np.array([line.split() for line in file], dtype=np.float32)
                    rel_x, rel_y = tmp_ne_feat[0, 0], tmp_ne_feat[0, 1]
                    tmp_ne_feat[:, 0] = tmp_ne_feat[:, 0] / rel_x
                    tmp_ne_feat[:, 1] = tmp_ne_feat[:, 1] / rel_y
                    x_train.append(tmp_ne_feat.flatten())
                    for e, emo in enumerate(self.emotions):
                        emo_lnd_path = os.path.join(landmarks_path, file_id + emo + self.profile[0] + '.txt')
                        with open(emo_lnd_path, 'r') as file:
                            tmp_emo_feat = np.array([line.split() for line in file], dtype=np.float32)
                        tmp_emo_feat[:, 0] = tmp_emo_feat[:, 0] / rel_x
                        tmp_emo_feat[:, 1] = tmp_emo_feat[:, 1] / rel_y
                        y_tmp.append(tmp_emo_feat.flatten())
                    y_train.append(y_tmp)
                    y_tmp = []
        np.save('./datasets/KDEF/input_landmarks_rel_train.npy', np.array(x_train))
        np.save('./datasets/KDEF/output_landmarks_rel_train.npy', np.array(y_train))

    def transform_NDPoints(self, trans_from, trans_to, src_trans):
        m, _ = cv2.estimateAffinePartial2D(trans_from, trans_to)
        dst = cv2.transform(src_trans[np.newaxis], m)
        return np.squeeze(dst)


class RafdData(EmotionalSpeechDataset):
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

    def rafd_frontal_faces(self):
        self.load_metadata_csv()
        rafd_frontal_path_list = self.metadata.loc[(self.metadata['gaze'] == 'frontal') &
                                                   (self.metadata['profile'] == 90)]['file_path'].tolist()
        with open('./datasets/Rafd/rafd_frontal_path_list.txt', 'w') as f:
            for fpath in rafd_frontal_path_list:
                f.write('{}\n'.format(fpath))


if __name__ == "__main__":
    rafd = RafdData('RafD')
    rafd.create_metadata_csv()
