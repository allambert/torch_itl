import os
import json
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
random.seed(21)
import shutil
import argparse

# structure for ImageFolder pytorch dataset
# DatasetName_Classification/train/class/file.png or
# DatasetName_Classification/test/class/file.png

base_data_path = ''
base_output_path = ''

KDEF_LABELS = ['AF', 'AN', 'DI', 'HA', 'NE', 'SA', 'SU']
Rafd_LABELS = ['fearful', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']


class EdgeMap(object):
    def __init__(self, out_res, num_parts=3):
        self.out_res = out_res
        self.num_parts = num_parts
        self.groups = [
            [np.arange(0, 17, 1), 255],
            [np.arange(17, 22, 1), 255],
            [np.arange(22, 27, 1), 255],
            [np.arange(27, 31, 1), 255],
            [np.arange(31, 36, 1), 255],
            [list(np.arange(36, 42, 1)) + [36], 255],
            [list(np.arange(42, 48, 1)) + [42], 255],
            [list(np.arange(48, 60, 1)) + [48], 255],
            [list(np.arange(60, 68, 1)) + [60], 255]
        ]

    def __call__(self, shape):
        image = np.zeros((self.out_res, self.out_res, self.num_parts), dtype=np.float32)
        for g in self.groups:
            for i in range(len(g[0]) - 1):
                start = int(shape[g[0][i]][0]), int(shape[g[0][i]][1])
                end = int(shape[g[0][i + 1]][0]), int(shape[g[0][i + 1]][1])
                cv2.line(image, start, end, g[1], 1)
        return image
        # return image.transpose(2, 0, 1) / 255.0


def FormatLndImageFolder(data_csv_name, lnd_folder, dataset_name, test_split):

    # get EdgeMap
    EM = EdgeMap(out_res=128, num_parts=1)

    # set directory structure
    output_path = os.path.join(base_output_path, dataset_name + '_LandmarkClassification')
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # read csv
    data_csv_path = os.path.join(base_data_path, data_csv_name)
    df = pd.read_csv(data_csv_path)

    # get and set train/test ids
    all_ids = df['speaker'].unique().tolist()
    test_ids = random.sample(all_ids, int(test_split*len(all_ids)))
    train_ids = list(set(all_ids) - set(test_ids))

    all_emotions = df['emotion'].unique().tolist()
    # for each emotion
    for emo in all_emotions:
        # get train test ids
        train_emo_spk = df.loc[(df['emotion'] == emo) & df['speaker'].isin(train_ids)]['file_path'].tolist()
        test_emo_spk = df.loc[(df['emotion']==emo) & df['speaker'].isin(test_ids)]['file_path'].tolist()

        # create dirs
        train_emo_path = os.path.join(train_path, emo)
        test_emo_path = os.path.join(test_path, emo)
        if not os.path.exists(train_emo_path):
            os.mkdir(train_emo_path)
        if not os.path.exists(test_emo_path):
            os.mkdir(test_emo_path)
        for row in train_emo_spk:
            lnd_file_basename = os.path.basename(row).split('.')[0]
            lnd_file_path = os.path.join(lnd_folder, lnd_file_basename + '.txt')
            lnd = np.loadtxt(lnd_file_path). reshape(68,2)
            lnd_img = EM(lnd)
            cv2.imwrite(os.path.join(train_emo_path, lnd_file_basename + '.JPG'), lnd_img)
        for row in test_emo_spk:
            lnd_file_basename = os.path.basename(row).split('.')[0]
            lnd_file_path = os.path.join(lnd_folder, lnd_file_basename + '.txt')
            lnd = np.loadtxt(lnd_file_path).reshape(68, 2)
            lnd_img = EM(lnd)
            cv2.imwrite(os.path.join(test_emo_path, lnd_file_basename + '.JPG'), lnd_img)

    with open(os.path.join(output_path, 'train_test_ids.csv'), 'w') as f:
        json.dump({'train_ids': train_ids, 'test_ids': test_ids}, f)


def FormatLndITLImageFolder(test_neu_folder, emo_lnd_folder, out_folder, dataset_name):
    EM = EdgeMap(out_res=128, num_parts=1)
    prefix_string = 'pred_'
    # read emotion names from directory names
    all_emotions = [f.name for f in os.scandir(os.path.dirname(test_neu_folder)) if f.is_dir()]
    print(prefix_string, all_emotions)

    # for each
    for fname in os.listdir(test_neu_folder):
        filename, file_extension = os.path.splitext(fname)
        for emo in all_emotions:
            # make output dir if does not exist
            emo_out_path = os.path.join(out_folder, emo)

            if not os.path.exists(emo_out_path):
                os.makedirs(emo_out_path)
            if dataset_name == 'KDEF':
                fname_emo = prefix_string + filename[0:4] + emo + 'S'
            elif dataset_name == 'Rafd':
                filename_split = filename.split('_')
                filename_split[4] = emo
                fname_emo = prefix_string + '_'.join(filename_split)
            elif dataset_name == 'RafdwoCON':
                filename_split = filename.split('_')
                filename_split[4] = emo
                fname_emo = prefix_string + '_'.join(filename_split)
            emo_lnd_file = os.path.join(emo_lnd_folder, fname_emo + '.txt')
            lnd = np.loadtxt(emo_lnd_file).reshape(68, 2)
            lnd_img = EM(lnd)
            cv2.imwrite(os.path.join(emo_out_path, fname_emo + '.JPG'), lnd_img)


def FormatLndITLKfold(test_neu_folder, emo_lnd_folder, out_folder, dataset_name, dirname_protocol):
    EM = EdgeMap(out_res=128, num_parts=1)
    prefix_string = 'pred_'
    # read emotion names from directory names
    all_emotions = [f.name for f in os.scandir(os.path.dirname(test_neu_folder)) if f.is_dir()]
    print(prefix_string, all_emotions)
    kdef_to_rafd = {key: Rafd_LABELS[i] for i, key in enumerate(KDEF_LABELS)}
    rafd_to_kdef = {key: KDEF_LABELS[i] for i, key in enumerate(Rafd_LABELS)}

    # for each
    for fname in os.listdir(emo_lnd_folder):
        filename = fname.split(prefix_string)[-1]
       # get emo
        if dataset_name == 'KDEF':
            emo = filename[4:6]
        elif dataset_name == 'Rafd':
            filename_split = filename.split('_')
            emo = filename_split[4]
        elif dataset_name == 'RafdwoCON':
            filename_split = filename.split('_')
            emo = filename_split[4]

        # make output dir if does not exist
        if dirname_protocol == dataset_name:
            emo_out_path = os.path.join(out_folder, emo)
        elif dataset_name == 'KDEF' and dirname_protocol == 'Rafd':
            emo_out_path = os.path.join(out_folder, kdef_to_rafd[emo])
        elif dataset_name == 'Rafd' and dirname_protocol == 'KDEF':
            emo_out_path = os.path.join(out_folder, rafd_to_kdef[emo])
        if not os.path.exists(emo_out_path):
            os.makedirs(emo_out_path)

        emo_lnd_file = os.path.join(emo_lnd_folder, fname)
        lnd = np.loadtxt(emo_lnd_file).reshape(68, 2)
        lnd_img = EM(lnd)
        cv2.imwrite(os.path.join(emo_out_path, fname.split('.')[0] + '.JPG'), lnd_img)


def FormatITLMultiFolder(test_neu_folder, emo_lnd_folder, out_folder, dataset_name, dirname_protocol):
    EM = EdgeMap(out_res=128, num_parts=1)
    # read emotion names from directory names
    all_emotions = [f.name for f in os.scandir(os.path.dirname(test_neu_folder)) if f.is_dir()]
    print(all_emotions)
    kdef_to_rafd = {key: Rafd_LABELS[i] for i, key in enumerate(KDEF_LABELS)}
    rafd_to_kdef = {key: KDEF_LABELS[i] for i, key in enumerate(Rafd_LABELS)}

    for dirpath, dirnames, filenames in os.walk(emo_lnd_folder):
        print(dirpath, dirnames)
        if dirpath.split('/')[-1] in all_emotions:
            # emo_out_path = os.path.join(out_folder, dirpath.split('/')[-1])
            # if not os.path.exists(emo_out_path):
            #     os.makedirs(emo_out_path)
            if dirname_protocol == dataset_name:
                emo_out_path = os.path.join(out_folder, dirpath.split('/')[-1])
            elif dataset_name == 'KDEF' and dirname_protocol == 'Rafd':
                emo_out_path = os.path.join(out_folder, kdef_to_rafd[dirpath.split('/')[-1]])
            elif dataset_name == 'Rafd' and dirname_protocol == 'KDEF':
                emo_out_path = os.path.join(out_folder, rafd_to_kdef[dirpath.split('/')[-1]])
            if not os.path.exists(emo_out_path):
                os.makedirs(emo_out_path)

            for f in filenames:
                if f.endswith('.txt'):
                    emo_lnd_file = os.path.join(dirpath, f)
                    lnd = np.loadtxt(emo_lnd_file).reshape(68, 2)
                    lnd_img = EM(lnd)
                    cv2.imwrite(os.path.join(emo_out_path, f.split('.')[0] + '.JPG'), lnd_img)


def FormatAffectnetEM(csv_path, out_folder='Aff_LandmarkClassification', num_samples_per_class=50):
    # get facealigner and EM
    #from utils.landmark_utils.facealigner import FaceAligner
    from ..facealigner import FaceAligner
    fa = FaceAligner(desiredFaceWidth=128)
    EM = EdgeMap(out_res=128, num_parts=1)

    # read csv
    df = pd.read_csv(csv_path, header=None)

    emo_dict = {0: 'Neutral',
                1: 'Happy',
                2: 'Sad',
                3: 'Surprise',
                4: 'Fear',
                5: 'Disgust',
                6: 'Anger'}

    # create output dir
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for key in emo_dict.keys():
        filter_df = df.loc[df[6] == key]
        emo_folder = os.path.join(out_folder, emo_dict[key])
        if not os.path.exists(emo_folder):
            os.mkdir(emo_folder)
        for i in range(num_samples_per_class):
            face_x = filter_df.iloc[i][1]
            face_y = filter_df.iloc[i][2]
            face_lnd = np.array([float(i) for i in filter_df.iloc[i][5].split(';')]).reshape((68,2))
            # set to origin
            face_lnd[:, 0] = face_lnd[:, 0] - face_x
            face_lnd[:, 1] = face_lnd[:, 1] - face_y
            # align and resize landmarks
            trans_face_lnd = fa.align(None, face_lnd)
            lnd_img = EM(trans_face_lnd)
            cv2.imwrite(os.path.join(emo_folder, str(i) + '.JPG'), lnd_img)



def FormatForImageFolder(data_csv_name, dataset_name, test_split):
    # set directory structure
    output_path = os.path.join(base_output_path, dataset_name + '_Classification')
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # read csv
    data_csv_path = os.path.join(base_data_path, data_csv_name)
    df = pd.read_csv(data_csv_path)

    # get and set train/test ids
    all_ids = df['speaker'].unique().tolist()
    test_ids = random.sample(all_ids, int(test_split*len(all_ids)))
    train_ids = list(set(all_ids) - set(test_ids))

    all_emotions = df['emotion'].unique().tolist()
    # for each emotion
    for emo in all_emotions:
        # get train test ids
        train_emo_spk = df.loc[(df['emotion'] == emo) & df['speaker'].isin(train_ids)]['file_path'].tolist()
        test_emo_spk = df.loc[(df['emotion']==emo) & df['speaker'].isin(test_ids)]['file_path'].tolist()

        # create dirs
        train_emo_path = os.path.join(train_path, emo)
        test_emo_path = os.path.join(test_path, emo)
        if not os.path.exists(train_emo_path):
            os.mkdir(train_emo_path)
        if not os.path.exists(test_emo_path):
            os.mkdir(test_emo_path)
        for row in train_emo_spk:
            shutil.copy2(row, os.path.join(train_emo_path, row.split('/')[-1]))
        for row in test_emo_spk:
            shutil.copy2(row, os.path.join(test_emo_path, row.split('/')[-1]))

    with open(os.path.join(output_path, 'train_test_ids.csv'), 'w') as f:
        json.dump({'train_ids': train_ids, 'test_ids': test_ids}, f)


def FormatSynthForImageFolder(test_neu_folder, test_lnd_folder, emo_lnd_folder,
                              out_folder, dataset_name, use_gt=True):

    prefix_string = '' if use_gt else 'pred_'
    # read emotion names from directory names
    all_emotions = [f.name for f in os.scandir(os.path.dirname(test_neu_folder)) if f.is_dir()]
    print(prefix_string, all_emotions)

    # for each
    for fname in os.listdir(test_neu_folder):
        filename, file_extension = os.path.splitext(fname)
        neu_img_file = os.path.join(test_neu_folder, fname)
        neu_lnd_file = os.path.join(test_lnd_folder, filename + '.txt')
        for emo in all_emotions:
            # make output dir if does not exist
            emo_out_path = os.path.join(out_folder, emo)
            if not os.path.exists(emo_out_path):
                os.makedirs(emo_out_path)
            if dataset_name == 'KDEF':
                fname_emo = prefix_string + filename[0:4] + emo + 'S'
            emo_lnd_file = os.path.join(emo_lnd_folder, fname_emo + '.txt')
            emo_out_file = os.path.join(emo_out_path, fname_emo + '.JPG')
            os.system('python /home/mlpboon/post-doc/repositories/torch_itl/utils/landmark_utils/'
                      'GANnotation/demo_gannotation.py ' +
                      ' '.join((neu_img_file, neu_lnd_file, emo_lnd_file, emo_out_file)))


if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument("--task", type=str, help="task_name")
    config.add_argument("--dataset_name", type=str, help="name of the dataset KDEF or Rafd")
    config.add_argument("--neu_img_folder", type=str, help="name of the neu_img_folder")
    config.add_argument("--emo_lnd_folder", type=str, help="name of the emo_lnd_folder")
    config.add_argument("--out_folder", type=str, help="name of the out_folder")
    config.add_argument("--dirname_protocol", type=str, help="label as rafd or kdef")

    args = config.parse_args()

    task = args.task #'edgemap'
    dataset_name = args.dataset_name #'Affectnet'

    if task == 'classify':
        if dataset_name == 'KDEF':
            data_csv_path = '../../../datasets/KDEF_Aligned/KDEF/KDEF.csv'
            FormatForImageFolder(data_csv_path, dataset_name, 0.1)
    elif task == 'synth':
        if dataset_name == 'KDEF':
            neu_img_folder = './KDEF_Classification/test/NE'
            neu_lnd_folder = '../../../datasets/KDEF_Aligned/KDEF_LANDMARKS'
            emo_lnd_folder = '../../../LS_Experiments/KDEF_relative_itl_model_20201121-181221/predictions/KDEF'
            out_folder = './SynthPredKDEF_relative_itl_model_20201121-181221'
            use_gt=False
            FormatSynthForImageFolder(neu_img_folder, neu_lnd_folder, emo_lnd_folder, out_folder, dataset_name, use_gt=use_gt)
    elif task == 'edgemap':
        if dataset_name == 'KDEF':
            lnd_folder = '../../../datasets/KDEF_Aligned/KDEF_LANDMARKS'
            data_csv_path = '../../../datasets/KDEF_Aligned/KDEF/KDEF.csv'
            FormatLndImageFolder(data_csv_path, lnd_folder, dataset_name, 0.1)
        elif dataset_name == 'Rafd':
            lnd_folder = '../../../datasets/Rafd_Aligned/Rafd_LANDMARKS'
            data_csv_path = '../../../datasets/Rafd_Aligned/Rafd/Rafd.csv'
            FormatLndImageFolder(data_csv_path, lnd_folder, dataset_name, 0.1)
        elif dataset_name == 'Affectnet':
            csv_path = '/media/mlpboon/X/AffectNetDatabase/training.csv'
            FormatAffectnetEM(csv_path, num_samples_per_class=2000)

    elif task == 'edgemapITL':
        if dataset_name == 'KDEF':
            neu_img_folder = './KDEF_LandmarkClassification/test/NE'
            emo_lnd_folder = '../../../LS_Experiments/KDEF_SU_itl_model_20201211-122024_CF/predictions/KDEF'
            out_folder = './EM_Classification_Exp/LndPredKDEF_SU_itl_model_20201211-122024_CF'
            FormatLndITLImageFolder(neu_img_folder, emo_lnd_folder, out_folder, dataset_name)
        elif dataset_name == 'Rafd':
            neu_img_folder = './Rafd_LandmarkClassification/test/neutral'
            emo_lnd_folder = '../../../LS_Experiments/Rafd_itl_model_20201118-134437/predictions/Rafd'
            out_folder = './LndPredRafd_itl_model_20201118-134437'
            FormatLndITLImageFolder(neu_img_folder, emo_lnd_folder, out_folder, dataset_name)
        elif dataset_name == 'RafdwoCON':
            neu_img_folder = './RafdwoCON_LandmarkClassification/test/neutral'
            emo_lnd_folder = '../../../LS_Experiments/Rafd_neutral_itl_model_20201203-180453/predictions/Rafd'
            out_folder = './LndPredRafd_neutral_itl_model_20201203-180453'
            FormatLndITLImageFolder(neu_img_folder, emo_lnd_folder, out_folder, dataset_name)
    elif task == 'edgemapITLkfold':
        neu_img_folder = args.neu_img_folder
        emo_lnd_folder = args.emo_lnd_folder
        out_folder = args.out_folder
        dirname_protocol = args.dirname_protocol
        FormatLndITLKfold(neu_img_folder, emo_lnd_folder, out_folder, dataset_name, dirname_protocol)
    elif task == 'edgemapITLJoint':
        neu_img_folder = args.neu_img_folder #'../../../LS_Experiments/KDEF_itl_model_20201210-171111/predictions/KDEF/NE'
        emo_lnd_folder = args.emo_lnd_folder #'../../../LS_Experiments/KDEF_itl_model_20201210-171111/predictions/KDEF'
        out_folder = args.out_folder  #'./LndPredJoint'
        dirname_protocol = args.dirname_protocol
        FormatITLMultiFolder(neu_img_folder, emo_lnd_folder, out_folder, dataset_name, dirname_protocol)