import os
import json
import pandas as pd
import random
import shutil
# structure for ImageFolder pytorch dataset
# DatasetName_Classification/train/class/file.png or
# DatasetName_Classification/test/class/file.png

base_data_path = ''
base_output_path = ''


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


if __name__ == "__main__":
    data_csv_path = 'KDEF.csv'
    dataset_name = 'KDEF'
    FormatForImageFolder(data_csv_path, dataset_name, 0.1)