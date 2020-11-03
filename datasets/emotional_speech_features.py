import numpy as np
import librosa
from .emotional_speech_datasets import Ravdess


def compute_mel_spectrogram(wav_paths_list, n_fft=1024, hop_length=256, n_mels=80, sr=22050, **kwargs):
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
        # read file in mono with specific sr
        x, _ = librosa.load(wav_file, sr=sr, mono=True)
        print(x.shape)
        # trim silences
        x_trim, ind_trim = librosa.effects.trim(x, frame_length=n_fft, hop_length=hop_length)
        print(x_trim.shape)
        # compute mel spec
        x_mel_spec = librosa.feature.melspectrogram(y=x_trim, sr=sr, power=1.0,
                                                    n_fft=n_fft, hop_length=hop_length,
                                                    n_mels=n_mels, **kwargs)
        assert x_mel_spec.shape[0] == n_mels
        x_out = dynamic_range_compression(x_mel_spec)
        n_features.append(x_out)
    return n_features


def dynamic_range_compression(x, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None))


def generate_training_samples(inp_list, context=5):
    training_sample_list = []
    for neu, emos in inp_list:
        print(neu, emos)
        neu_feat = compute_mel_spectrogram([neu])
        n_emos = len(emos)
        n_freq, n_frames = neu_feat[0].shape
        train_x = np.zeros((n_frames - (2*context), n_freq, (2*context) + 1))
        train_y = np.zeros((n_frames-2*context, n_emos, n_freq))
        # print(train_x.shape)
        # print(train_y.shape)

        for ne, emo in enumerate(emos):
            emo_feat = compute_mel_spectrogram([emo])
            # align sequences using dtw
            cost, wp = librosa.sequence.dtw(neu_feat[0], emo_feat[0])
            #print(wp)
            # Use the alignment to create training pairs
            # Neutral(F x N_(i-n: i+n)) --> Emotional(F)_(N_i)
            for i in range(context, n_frames-context):
                #print(i)
                if ne == 0:
                    train_x[i-context, :, :] = neu_feat[0][:, i-context:i+context+1]
                train_y[i-context, ne, :] = emo_feat[0][:, wp[wp[:, 0] == i, 1][0]]
        training_sample_list.append((train_x, train_y))
    return training_sample_list


if __name__ == '__main__':
    db_rds = Ravdess('RAVDESS')
    db_rds.load_metadata_csv()
    print(db_rds.speakers)
    set_speakers = ['Actor_01']
    set_emotions = ['neutral', 'sad', 'happy', 'calm']
    set_sentence = db_rds.sentence_dict['01']
    set_repetition = int(db_rds.repetition_dict['01'])
    db_rds_filtered = db_rds.get_speaker_emotion(set_speakers, set_emotions)
    train_db_rds = db_rds_filtered[(db_rds_filtered['sentence'] == set_sentence) &
                                   (db_rds_filtered['repetition'] == set_repetition)]
    inp_list = [[train_db_rds[train_db_rds['emotion'] == 'neutral']['file_path'].tolist()[0],
                train_db_rds[train_db_rds['emotion'] != 'neutral']['file_path'].tolist()]]
    training_samples = generate_training_samples(inp_list)
