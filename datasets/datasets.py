import numpy as np
import os
import torch
import pandas as pd

from .synthetic_quantile import toy_data_quantile
from .emotional_speech_datasets import Ravdess, KdefData
from .emotional_speech_features import generate_training_samples

np.random.seed(seed=21)
import random

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
    from sklearn.preprocessing import StandardScaler
    """
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
    neu_list = train_db_rds[train_db_rds['emotion'] == 'neutral']['file_path'].tolist()[0]
    emo_list = train_db_rds[train_db_rds['emotion'] != 'neutral']['file_path'].tolist()
    inp_list = [[neu_list, emo_list]]

    # TODO: Remove dirty workaround for order checking (needed for sampler at the moment)
    assert([emo.split('/')[-1] for emo in emo_list] == ['03-01-02-01-01-01-01.wav',
                                                        '03-01-03-01-01-01-01.wav',
                                                        '03-01-04-01-01-01-01.wav'])
    inp_list = [['./RAVDESS/Actor_01/03-01-01-01-01-01-01.wav',
                 ['./RAVDESS/Actor_01/03-01-02-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-03-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-04-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-05-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-08-01-01-01-01.wav']],
                ['./RAVDESS/Actor_01/03-01-01-01-02-01-01.wav',
                 ['./RAVDESS/Actor_01/03-01-02-01-02-01-01.wav',
                  './RAVDESS/Actor_01/03-01-03-01-02-01-01.wav',
                  './RAVDESS/Actor_01/03-01-04-01-02-01-01.wav',
                  './RAVDESS/Actor_01/03-01-05-01-02-01-01.wav',
                  './RAVDESS/Actor_01/03-01-08-01-02-01-01.wav']]
                ]
    """
    inp_list = [['./RAVDESS/Actor_01/03-01-01-01-01-01-01.wav',
                 ['./RAVDESS/Actor_01/03-01-06-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-05-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-07-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-03-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-04-01-01-01-01.wav',
                  './RAVDESS/Actor_01/03-01-08-01-01-01-01.wav']],
                ['./RAVDESS/Actor_01/03-01-01-01-02-01-01.wav',
                 ['./RAVDESS/Actor_01/03-01-06-01-02-01-01.wav',
                  './RAVDESS/Actor_01/03-01-05-01-02-01-01.wav',
                  './RAVDESS/Actor_01/03-01-07-01-02-01-01.wav',
                  './RAVDESS/Actor_01/03-01-03-01-02-01-01.wav',
                  './RAVDESS/Actor_01/03-01-04-01-02-01-01.wav',
                  './RAVDESS/Actor_01/03-01-08-01-02-01-01.wav']]
                ]
    training_samples = generate_training_samples(inp_list, context=0)
    # x_train = torch.from_numpy(np.expand_dims(training_samples[0][0], axis=1)).float()
    # x_train = torch.from_numpy(np.squeeze(training_samples[0][0])).float()
    # x_train = torch.from_numpy(np.squeeze(np.concatenate((training_samples[0][0], training_samples[1][0])))).float()
    # y_train = torch.from_numpy(np.concatenate((training_samples[0][1], training_samples[1][1]))).float()
    # y_train = torch.cat((y_train_emo, torch.unsqueeze(x_train, 1)), 1)

    x_train_npy = np.squeeze(np.concatenate((training_samples[0][0], training_samples[1][0])))
    scaler = StandardScaler(with_std=True)
    x_train_scaled = scaler.fit_transform(x_train_npy)
    x_mean = scaler.mean_
    x_train = torch.from_numpy(x_train_scaled).float()
    y_train = torch.from_numpy(np.concatenate((training_samples[0][1], training_samples[1][1]))).float()

    # m = y_train_npy.shape[1]
    # train for the difference
    # y_train = torch.from_numpy(y_train_npy - np.repeat(np.expand_dims(x_train_npy, axis=1), m, axis=1)).float()
    return x_train, y_train, x_mean


def import_toy_synthesis(n_samples, n_theta):
    x1 = np.linspace(start=-1, stop=1, num=n_samples)
    x2 = np.zeros(n_samples)
    np.random.shuffle(x1)
    X = np.vstack((x1, x2)).T

    thetas = np.linspace(start=-np.pi/2, stop=np.pi/2, num=n_theta)
    Y = np.zeros((n_samples, n_theta, 2))
    for i in range(n_samples):
        Y[i, :, 0] = x1[i]*np.cos(thetas)
        Y[i, :, 1] = np.abs(x1[i])*np.sin(thetas)

    x_train = X[0:n_samples//2]
    y_train = Y[0:n_samples//2]
    x_test = X[n_samples//2:]
    y_test = Y[n_samples//2:]
    return torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), \
        torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()


def import_kdef_synthesis():
    X = np.load('./datasets/KDEF/input_train_affnet.npy')
    Y = np.load('./datasets/KDEF/output_train_affnet.npy')
    n_samples = len(X)
    x_train = np.concatenate((X[1:n_samples//2 - 1], X[n_samples//2 + 1:n_samples-1]))
    y_train = np.concatenate((Y[1:n_samples//2 - 1], Y[n_samples//2 + 1:n_samples-1]))
    x_test = X[np.array([0, n_samples//2 - 1, n_samples//2, n_samples - 1])]
    y_test = Y[np.array([0, n_samples//2 - 1, n_samples//2, n_samples - 1])]
    return torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), \
        torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()


def import_kdef_landmark_synthesis(dtype='aligned'):
    if dtype == 'aligned':
        X = np.load('./datasets/KDEF/input_landmarks.npy')
        Y = np.load('./datasets/KDEF/output_landmarks.npy')
        X = X / 562.
        Y = Y / 562.
    if dtype == 'aligned2':
        X = np.load('./datasets/KDEF/input_landmarks_align2.npy')
        Y = np.load('./datasets/KDEF/output_landmarks_align2.npy')
        X = X / 562.
        Y = Y / 562.
    elif dtype == 'diff':
        X = np.load('./datasets/KDEF/input_landmarks.npy')
        Y_tmp = np.load('./datasets/KDEF/output_landmarks.npy')
        m = Y_tmp.shape[1]
        X = X / 562.
        Y_tmp = Y_tmp / 562.
        Y = Y_tmp - np.repeat(np.expand_dims(X, axis=1), m, axis=1)
    else:
        print('No such data type exists')
    n_samples = len(X)
    x_train = np.concatenate((X[1:n_samples//2 - 1], X[n_samples//2 + 1:n_samples-1]))
    y_train = np.concatenate((Y[1:n_samples//2 - 1], Y[n_samples//2 + 1:n_samples-1]))
    x_test = X[np.array([0, n_samples//2 - 1, n_samples//2, n_samples - 1])]
    y_test = Y[np.array([0, n_samples//2 - 1, n_samples//2, n_samples - 1])]
    return torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), \
        torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()


def kdef_landmarks_facealigner(path_to_landmarks, inp_emotion='NE', inc_emotion=False, relative=False, kfold=0,
                               random_seed=21):
    # init lists
    train_list = []
    test_list = []

    # generate ids + emotion lists (dataloader not trusted here)
    fem_ids = ['F'+str(i).zfill(2) for i in range(1, 36)]
    mal_ids = ['M'+str(i).zfill(2) for i in range(1, 36)]
    all_ids = fem_ids + mal_ids


    # set test identities
    if kfold == 0:
        # test_identities = ['F01', 'F02', 'F03', 'M01', 'M02', 'M03']
        test_identities = ["F22", "M19", "M34", "M02", "M27", "F28", "M26"]
    else:
        shuffle_all_ids = all_ids.copy()
        random.Random(random_seed).shuffle(shuffle_all_ids)
        test_num = len(shuffle_all_ids)//10  # 90/10 split
        if kfold <= len(shuffle_all_ids)//test_num:
            test_identities = shuffle_all_ids[(kfold-1)*test_num:kfold*test_num]
        else:
            test_identities = shuffle_all_ids[(kfold-1)*test_num:]

    # define emotion list, same as in sampler (different abbrv. due to dataset)
    all_emotions = ['AF', 'AN', 'DI', 'HA', 'SA', 'SU', 'NE']
    # find inp emotion index
    inp_emo_idx = all_emotions.index(inp_emotion)

    for sess in ['A', 'B']:
        for p in all_ids:
            # create file list for all emos
            file_list = [sess + p + em + 'S.txt' for em in all_emotions]
            if inc_emotion:
                # if inc_emotion is True just include evth as target
                xy_pair = [file_list[inp_emo_idx], file_list]
            else:
                # if not pop inp_emotion
                inp_emotion_file = file_list.pop(inp_emo_idx)
                xy_pair = [inp_emotion_file, file_list]
            if p not in test_identities:
                train_list.append(xy_pair)
            else:
                test_list.append(xy_pair)

    train_input = []
    train_output = []
    test_input = []
    test_output = []

    # read input/output train
    for row in train_list:
        tmp_ne = np.loadtxt(os.path.join(path_to_landmarks, row[0])).reshape(68, 2)
        train_input.append(tmp_ne.flatten())
        tmp_ems = []
        for row_em in row[1]:
            tmp_em = np.loadtxt(os.path.join(path_to_landmarks, row_em)).reshape(68, 2)
            tmp_ems.append(tmp_em.flatten())
        train_output.append(tmp_ems)

    # read input/output test
    for row in test_list:
        tmp_ne = np.loadtxt(os.path.join(path_to_landmarks, row[0])).reshape(68,2)
        test_input.append(tmp_ne.flatten())
        tmp_ems = []
        for row_em in row[1]:
            tmp_em = np.loadtxt(os.path.join(path_to_landmarks, row_em)).reshape(68, 2)
            tmp_ems.append(tmp_em.flatten())
        test_output.append(tmp_ems)


    im_size = 128
    # normalize between [-1,1]
    # train_input = (2 * np.array(train_input)/im_size) - 1
    # train_output = (2 * np.array(train_output) / im_size) - 1
    # test_input = (2 * np.array(test_input) / im_size) - 1
    # test_output = (2 * np.array(test_output) / im_size) - 1

    # normalize between [0,1]
    train_input = np.array(train_input)/im_size
    train_output = np.array(train_output) / im_size
    test_input = np.array(test_input) / im_size
    test_output = np.array(test_output) / im_size

    # np.save('facealigner_train_input.npy', train_input)
    # np.save('facealigner_train_output.npy', train_output)
    # np.save('facealigner_test_input.npy', test_input)
    # np.save('facealigner_test_output.npy', test_output)

    if relative:
        m = train_output.shape[1]
        train_output = train_output - np.repeat(np.expand_dims(train_input, axis=1), m, axis=1)
        test_output = test_output - np.repeat(np.expand_dims(test_input, axis=1), m, axis=1)
    return torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float(),\
           torch.from_numpy(test_input).float(), torch.from_numpy(test_output).float(), \
           train_list, test_list


def kdef_landmarks_facenet(path_to_landmarks, path_to_emb, inp_emotion='NE', inc_emotion=False, relative=False):
    # init lists
    train_list = []
    test_list = []

    # generate ids + emotion lists (dataloader not trusted here)
    fem_ids = ['F'+str(i).zfill(2) for i in range(1, 36)]
    mal_ids = ['M'+str(i).zfill(2) for i in range(1, 36)]
    all_ids = fem_ids + mal_ids


    # set test identities
    # test_identities = ['F01', 'F02', 'F03', 'M01', 'M02', 'M03']
    test_identities = ["F22", "M19", "M34", "M02", "M27", "F28", "M26"]

    # define emotion list, same as in sampler (different abbrv. due to dataset)
    all_emotions = ['AF', 'AN', 'DI', 'HA', 'SA', 'SU', 'NE']
    # find inp emotion index
    inp_emo_idx = all_emotions.index(inp_emotion)

    for sess in ['A', 'B']:
        for p in all_ids:
            # create file list for all emos
            file_list = [sess + p + em + 'S.txt' for em in all_emotions]
            if inc_emotion:
                # if inc_emotion is True just include evth as target
                xy_pair = [file_list[inp_emo_idx], file_list]
            else:
                # if not pop inp_emotion
                inp_emotion_file = file_list.pop(inp_emo_idx)
                xy_pair = [inp_emotion_file, file_list]
            if p not in test_identities:
                train_list.append(xy_pair)
            else:
                test_list.append(xy_pair)

    train_input = []
    train_output = []
    test_input = []
    test_output = []

    # read input/output train
    for row in train_list:
        tmp_ne = np.load(os.path.join(path_to_emb, row[0].split('.')[0] + '.npy'))
        train_input.append(tmp_ne.flatten())
        tmp_ems = []
        for row_em in row[1]:
            tmp_em = np.loadtxt(os.path.join(path_to_landmarks, row_em)).reshape(68, 2)
            tmp_ems.append(tmp_em.flatten())
        train_output.append(tmp_ems)

    # read input/output test
    for row in test_list:
        tmp_ne = np.load(os.path.join(path_to_emb, row[0].split('.')[0] + '.npy'))
        test_input.append(tmp_ne.flatten())
        tmp_ems = []
        for row_em in row[1]:
            tmp_em = np.loadtxt(os.path.join(path_to_landmarks, row_em)).reshape(68, 2)
            tmp_ems.append(tmp_em.flatten())
        test_output.append(tmp_ems)


    im_size = 128
    # normalize between [-1,1]
    # train_input = (2 * np.array(train_input)/im_size) - 1
    # train_output = (2 * np.array(train_output) / im_size) - 1
    # test_input = (2 * np.array(test_input) / im_size) - 1
    # test_output = (2 * np.array(test_output) / im_size) - 1

    # normalize between [0,1]
    train_input = np.array(train_input)
    train_output = np.array(train_output) / im_size
    test_input = np.array(test_input)
    test_output = np.array(test_output) / im_size

    # np.save('facealigner_train_input.npy', train_input)
    # np.save('facealigner_train_output.npy', train_output)
    # np.save('facealigner_test_input.npy', test_input)
    # np.save('facealigner_test_output.npy', test_output)

    return torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float(),\
           torch.from_numpy(test_input).float(), torch.from_numpy(test_output).float(), \
           train_list, test_list


def kdef_landmarks_id_emo(path_to_landmarks, path_to_emb, inp_emotion='NE', inc_emotion=True, relative=False):
    ### BIG WARNING!!!! ####
    print('This function is currently hardcoded for NE input emotion, BE AWARE')

    # init lists
    train_list = []
    test_list = []

    emo_va = {'NE': [0, 0],
              'HA': [0.6647930758154823, 0.07025315235958239],
              'SA': [-0.6364936709598201, -0.25688320447600566],
              'SU': [0.17960005233680493, 0.6894792038743631],
              'AF': [-0.1253752865553082, 0.7655788112100937],
              'DI': [-0.6943645673378837, 0.457145871269001],
              'AN': [-0.452336028803629, 0.5656012294430937],
              }

    # generate ids + emotion lists (dataloader not trusted here)
    fem_ids = ['F'+str(i).zfill(2) for i in range(1, 36)]
    mal_ids = ['M'+str(i).zfill(2) for i in range(1, 36)]
    all_ids = fem_ids + mal_ids


    # set test identities
    # test_identities = ['F01', 'F02', 'F03', 'M01', 'M02', 'M03']
    test_identities = ["F22", "M19", "M34", "M02", "M27", "F28", "M26"]

    # define emotion list, same as in sampler (different abbrv. due to dataset)
    all_emotions = ['AF', 'AN', 'DI', 'HA', 'SA', 'SU', 'NE']
    # find inp emotion index
    inp_emo_idx = all_emotions.index(inp_emotion)

    for sess in ['A', 'B']:
        for p in all_ids:
            # create file list for all emos
            file_list = [sess + p + em + 'S.txt' for em in all_emotions]
            if inc_emotion:
                # if inc_emotion is True just include evth as target
                xy_pair = [file_list[inp_emo_idx], file_list]
            else:
                # if not pop inp_emotion
                inp_emotion_file = file_list.pop(inp_emo_idx)
                xy_pair = [inp_emotion_file, file_list]
            if p not in test_identities:
                train_list.append(xy_pair)
            else:
                test_list.append(xy_pair)

    train_input = []
    train_output = []
    test_input = []
    test_output = []

    # read input/output train
    for row in train_list:
        tmp_ne = np.load(os.path.join(path_to_emb, row[0].split('.')[0] + '.npy'))
        train_input.append(np.concatenate([tmp_ne.flatten(), emo_va['NE']]))
        tmp_ems = []
        for i, row_em in enumerate(row[1]):
            tmp_em = np.loadtxt(os.path.join(path_to_landmarks, row_em)).reshape(68, 2)
            if 'NE' in row_em:
                tmp_ems.append(np.concatenate([tmp_em.flatten(), emo_va[all_emotions[i]]]))
            else:
                tmp_ems.append(np.concatenate([tmp_em.flatten(),
                                           emo_va[all_emotions[i]]/np.linalg.norm(emo_va[all_emotions[i]])]))
        train_output.append(tmp_ems)

    # read input/output test
    for row in test_list:
        tmp_ne = np.load(os.path.join(path_to_emb, row[0].split('.')[0] + '.npy'))
        test_input.append(np.concatenate([tmp_ne.flatten(), emo_va['NE']]))
        tmp_ems = []
        for i, row_em in enumerate(row[1]):
            tmp_em = np.loadtxt(os.path.join(path_to_landmarks, row_em)).reshape(68, 2)
            if 'NE' in row_em:
                tmp_ems.append(np.concatenate([tmp_em.flatten(), emo_va[all_emotions[i]]]))
            else:
                tmp_ems.append(np.concatenate([tmp_em.flatten(),
                                               emo_va[all_emotions[i]] / np.linalg.norm(emo_va[all_emotions[i]])]))
        test_output.append(tmp_ems)


    im_size = 128
    # normalize between [-1,1]
    # train_input = (2 * np.array(train_input)/im_size) - 1
    # train_output = (2 * np.array(train_output) / im_size) - 1
    # test_input = (2 * np.array(test_input) / im_size) - 1
    # test_output = (2 * np.array(test_output) / im_size) - 1

    # normalize between [0,1]
    train_input = np.array(train_input)
    train_output = np.array(train_output) / im_size
    test_input = np.array(test_input)
    test_output = np.array(test_output) / im_size

    # np.save('facealigner_train_input.npy', train_input)
    # np.save('facealigner_train_output.npy', train_output)
    # np.save('facealigner_test_input.npy', test_input)
    # np.save('facealigner_test_output.npy', test_output)

    return torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float(),\
           torch.from_numpy(test_input).float(), torch.from_numpy(test_output).float(), \
           train_list, test_list




def rafd_landmarks_facealigner(path_to_landmarks, path_to_csv, inp_emotion='neutral', inc_emotion=False, kfold=0,
                               random_seed=21):
    train_list = []
    test_list = []
    all_ids = [str(i).zfill(2) for i in pd.read_csv(path_to_csv)['speaker'].unique().tolist()]

    # set test identities
    if kfold == 0:
        test_ids = ['25', '58', '72', '41', '35', '71']
    else:
        shuffle_all_ids = all_ids.copy()
        random.Random(random_seed).shuffle(shuffle_all_ids)
        test_num = len(shuffle_all_ids) // 10  # 90/10 split
        if kfold <= len(shuffle_all_ids) // test_num:
            test_ids = shuffle_all_ids[(kfold - 1) * test_num:kfold * test_num]
        else:
            test_ids = shuffle_all_ids[(kfold - 1) * test_num:]

    all_emotions = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    inp_emo_idx = all_emotions.index(inp_emotion)
    num_emos = len(all_emotions)
    #print(num_emos)
    sorted_lnd_list = sorted(os.listdir(path_to_landmarks))

    for i in range(len(all_ids)):
        fnames = sorted_lnd_list[i*num_emos: (i+1)*num_emos]
        # get neutral and put it at the end
        neu_im = fnames.pop(5)
        fnames.append(neu_im)

        #woCON change 1 of 3
        con_im = fnames.pop(1)
        #print(con_im)
        #woCON change

        # print(fnames)
        # print(all_ids[i])
        assert set([fn[8:10] for fn in fnames]) == set([all_ids[i]])
        # woCON change 2 of 3
        assert set([fn.split('_')[4] for fn in fnames]) == set(all_emotions)- {'contemptuous'}
        if inp_emo_idx > 1:
            inp_im = fnames[inp_emo_idx-1]
        else:
            inp_im = fnames[inp_emo_idx]
        # woCON change
        if not inc_emotion:
            fnames.remove(inp_im)
        if fnames[0][8:10] not in test_ids:
            train_list.append([inp_im, fnames])
        else:
            test_list.append([inp_im, fnames])

    train_input = []
    train_output = []
    test_input = []
    test_output = []

    # read input/output train
    for row in train_list:
        tmp_ne = np.loadtxt(os.path.join(path_to_landmarks, row[0])).reshape(68, 2)
        train_input.append(tmp_ne.flatten())
        tmp_ems = []
        for row_em in row[1]:
            tmp_em = np.loadtxt(os.path.join(path_to_landmarks, row_em)).reshape(68, 2)
            tmp_ems.append(tmp_em.flatten())
        train_output.append(tmp_ems)

    # read input/output test
    for row in test_list:
        tmp_ne = np.loadtxt(os.path.join(path_to_landmarks, row[0])).reshape(68,2)
        test_input.append(tmp_ne.flatten())
        tmp_ems = []
        for row_em in row[1]:
            tmp_em = np.loadtxt(os.path.join(path_to_landmarks, row_em)).reshape(68, 2)
            tmp_ems.append(tmp_em.flatten())
        test_output.append(tmp_ems)
    im_size = 128
    # normalize between [0,1]
    train_input = np.array(train_input) / im_size
    train_output = np.array(train_output) / im_size
    test_input = np.array(test_input) / im_size
    test_output = np.array(test_output) / im_size
    return torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float(), \
           torch.from_numpy(test_input).float(), torch.from_numpy(test_output).float(), \
           train_list, test_list


def import_affectnet_va_embedding(affect_net_csv_path):
    if not affect_net_csv_path == '':
        df = pd.read_csv(affect_net_csv_path, header=None)
        emo_dict = {0: 'Neutral',
                    1: 'Happy',
                    2: 'Sad',
                    3: 'Surprise',
                    4: 'Fear',
                    5: 'Disgust',
                    6: 'Anger',
                    7: 'Contempt'}
        emo_va = {}
        for key in emo_dict.keys():
            emo_va[emo_dict[key]] = [df[df[6] == key][7].mean(),
                                     df[df[6] == key][8].mean()]
    else:
        emo_va = {'Neutral': [-0.06249655698883391, -0.019669702033559486],
         'Happy': [0.6647930758154823, 0.07025315235958239],
         'Sad': [-0.6364936709598201, -0.25688320447600566],
         'Surprise': [0.17960005233680493, 0.6894792038743631],
         'Fear': [-0.1253752865553082, 0.7655788112100937],
         'Disgust': [-0.6943645673378837, 0.457145871269001],
         'Anger': [-0.452336028803629, 0.5656012294430937],
         'Contempt': [-0.5138537435929467, 0.5825553992724378]}
    return emo_va
