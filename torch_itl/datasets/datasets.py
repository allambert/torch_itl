import os
import torch
import numpy as np
np.random.seed(seed=21)
import random
from .synthetic_quantile import toy_data_quantile


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


def kdef_landmarks_facealigner(path_to_landmarks, inp_emotion='NE', inc_emotion=False,
                               kfold=0, random_seed=21):
    """
    Get data to train vITL for KDEF
    Parameters
    ----------
    path_to_landmarks: str
                       path to folder containing landmarks
    inp_emotion: str
                 the input emotion for single model
    inc_emotion: bool
                 whether to include inp_emotion in output
    kfold: int
           which split to compute data for, 0 is a preselected split, start from 1
    random_seed: set to 21 for reproducibility

    Returns
    -------
    x_train, y_train, x_test, y_test, train_list, test_list
    """
    # init lists
    train_list = []
    test_list = []

    # generate ids + emotion lists (dataloader not trusted here)
    fem_ids = ['F'+str(i).zfill(2) for i in range(1, 36)]
    mal_ids = ['M'+str(i).zfill(2) for i in range(1, 36)]
    all_ids = fem_ids + mal_ids

    # set test identities
    if kfold == 0:
        test_identities = ["F22", "M19", "M34", "M02", "M27", "F28", "M26"]
    else:
        shuffle_all_ids = all_ids.copy()
        random.Random(random_seed).shuffle(shuffle_all_ids)
        test_num = len(shuffle_all_ids)//10  # 90/10 split
        test_identities = shuffle_all_ids[(kfold - 1) * test_num:kfold * test_num]

    # define emotion list, same as in sampler (different abbrv. due to dataset)
    all_emotions = ['AN', 'DI', 'AF', 'HA', 'SA', 'SU', 'NE']
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
    # normalize between [0,1]
    train_input = np.array(train_input)/im_size
    train_output = np.array(train_output) / im_size
    test_input = np.array(test_input) / im_size
    test_output = np.array(test_output) / im_size

    return torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float(),\
           torch.from_numpy(test_input).float(), torch.from_numpy(test_output).float(), \
           train_list, test_list


def rafd_landmarks_facealigner(path_to_landmarks, inp_emotion='neutral', inc_emotion=False,
                               kfold=0, random_seed=21):
    """
    Get data to train vITL for RaFD

    Parameters
    ----------
    path_to_landmarks: str
                       path to folder cotaining landmarks
    inp_emotion: str
                 the input emotion for single model
    inc_emotion: bool
                 whether to include inp_emotion in output
    kfold: int
           which split to compute data for, 0 is a preselected split, start from 1
    random_seed: set to 21 for reproducibility

    Returns
    -------
    x_train, y_train, x_test, y_test, train_list, test_list
    """

    train_list = []
    test_list = []
    all_ids = ['01', '02', '03', '04', '05', '07', '08', '09', '10', '11', '12',
               '14', '15', '16', '18', '19', '20', '21', '22', '23', '24', '25',
               '26', '27', '28', '29', '30', '31', '32', '33', '35', '36', '37',
               '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48',
               '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
               '60', '61', '63', '64', '65', '67', '68', '69', '70', '71', '72', '73']

    # set test identities
    if kfold == 0:
        test_ids = ['25', '58', '72', '41', '35', '71']
    else:
        shuffle_all_ids = all_ids.copy()
        random.Random(random_seed).shuffle(shuffle_all_ids)
        test_num = len(shuffle_all_ids) // 10  # 90/10 split
        test_ids = shuffle_all_ids[(kfold - 1) * test_num:kfold * test_num]

    all_emotions = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    inp_emo_idx = all_emotions.index(inp_emotion)
    num_emos = len(all_emotions)
    sorted_lnd_list = sorted(os.listdir(path_to_landmarks))

    for i in range(len(all_ids)):
        fnames = sorted_lnd_list[i*num_emos: (i+1)*num_emos]
        # get neutral and put it at the end
        neu_im = fnames.pop(5)
        fnames.append(neu_im)
        con_im = fnames.pop(1)

        assert set([fn[8:10] for fn in fnames]) == set([all_ids[i]])
        assert set([fn.split('_')[4] for fn in fnames]) == set(all_emotions)- {'contemptuous'}

        if inp_emo_idx > 1:
            inp_im = fnames[inp_emo_idx-1]
        else:
            inp_im = fnames[inp_emo_idx]

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


def get_data_landmarks(dataset, path_to_landmarks, kfold=0, random_seed=21):
    '''Loader for both KDEF and RaFD datasets
    Loads all available data
    Parameters
    ----------
    dataset: str
        'KDEF' or 'RaFD'
    path_to_landmarks: str
                       path to folder cotaining landmarks
    kfold: int
           which split to compute data for, 0 is a preselected split, start from 1
    random_seed: set to 21 for reproducibility

    Returns
    -------
    data_train, data_test
    '''
    if dataset=='KDEF':
        x_train, y_train, x_test, y_test, train_list, test_list = kdef_landmarks_facealigner(
            path_to_landmarks, inp_emotion='AN', inc_emotion=True, kfold=kfold, random_seed=random_seed)

    elif dataset=='RaFD':
        x_train, y_train, x_test, y_test, train_list, test_list = rafd_landmarks_facealigner(
            path_to_landmarks, inp_emotion='angry', inc_emotion=True, kfold=kfold, random_seed=random_seed)

    else:
        raise Exception('Unknown dataset')

    return y_train, y_test

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
