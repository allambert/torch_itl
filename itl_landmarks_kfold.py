import os
import subprocess
import json
import time


def plot_box(plot_dict, dataset, itl_type, itl_out_folder):
    import pandas as pd
    import matplotlib.pyplot as plt

    itl_result_plot_dir = os.path.join(itl_out_folder, 'results_and_plots')

    if not os.path.exists(itl_result_plot_dir):
        os.mkdir(itl_result_plot_dir)
    with open(os.path.join(itl_result_plot_dir, 'data.txt'), 'w') as outfile:
        json.dump(plot_dict, outfile, indent=4)

    for subkey in ['cost', 'accuracy']:
        sub_dict = {key: plot_dict[key][subkey] for key in plot_dict.keys()}
        df = pd.DataFrame.from_dict(sub_dict)
        df_plt = df.plot.box(grid=True)
        myfig = df_plt.get_figure()
        myfig.savefig(os.path.join(itl_result_plot_dir,
                                   '_'.join([dataset, itl_type, subkey])))

        print(df.mean(), df.std())


dataset = 'Rafd'
itl_type = 'single'

ClassificationBaseFolder = './utils/landmark_utils/Classification'

if dataset == 'KDEF':
    all_emotions = ['AF', 'AN', 'DI', 'HA', 'SA', 'SU', 'NE']
    #all_emotions = ['NE']
    num_fold = 10
    # itl out folder
    out_folder_name = dataset + '_' + itl_type
    itl_out_folder = './LS_Experiments/' + out_folder_name + '/'
    # format EM params
    test_neu_folder = os.path.join(ClassificationBaseFolder, './KDEF_LandmarkClassification/test/NE')
    format_out_folder = os.path.join(ClassificationBaseFolder, 'EM_Classification_Exp', out_folder_name)
    dirname_protocol = 'Rafd'

    # Classification params
    if dirname_protocol == dataset:
        model_path = os.path.join(ClassificationBaseFolder,
                                  './LndExperiments/KDEF_bs16_e10_20201117-181507')
    elif dirname_protocol == 'Rafd':
        model_path = os.path.join(ClassificationBaseFolder,
                                  './LndExperiments/RafdwoCON_bs16_e10_20201203-133925')

elif dataset == 'Rafd':
    all_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    #all_emotions = ['neutral']
    num_fold = 10
    # itl out folder
    out_folder_name = dataset + '_' + itl_type
    itl_out_folder = './LS_Experiments/' + out_folder_name + '/'
    # format EM params
    test_neu_folder = os.path.join(ClassificationBaseFolder, './RafdwoCON_LandmarkClassification/test/neutral')
    format_out_folder = os.path.join(ClassificationBaseFolder, 'EM_Classification_Exp', out_folder_name)
    dirname_protocol = 'KDEF'

    # Classification params
    if dirname_protocol == dataset:
        model_path = os.path.join(ClassificationBaseFolder,
                                  './LndExperiments/RafdwoCON_bs16_e10_20201203-133925')
    elif dirname_protocol == 'KDEF':
        model_path = os.path.join(ClassificationBaseFolder,
                                  './LndExperiments/KDEF_bs16_e10_20201117-181507')


em_test_costs = {em: {'cost': [], 'accuracy': []} for em in all_emotions}
start_time = time.time()
if itl_type == 'single':
    for em in all_emotions:
        for fold in range(1, num_fold+1):
            itl_cmd = ' '.join(['python itl_one_emotion_argparse.py',
                                '--dataset '+dataset,
                                '--input_emotion '+em,
                                '--inc_emotion',
                                '--kfold '+str(fold),
                                '--save_pred',
                                '--save_model',
                                '--output_folder '+itl_out_folder])
            with subprocess.Popen(itl_cmd, shell=True, stdout=subprocess.PIPE) as cmd:
                for line in cmd.stdout:
                    if "test cost" in line.decode("utf-8"):
                        em_test_costs[em]['cost'].append(float(line.decode("utf-8").split(' ')[-1].strip('\n')))

            # Format ITL to Edgemaps
            emo_lnd_folder = os.path.join(itl_out_folder, dataset + '_' + em + '_itl_model_' +
                            'split' + str(fold) + '_CF', 'predictions', dataset)
            format_itl_out = os.path.join(format_out_folder, dataset + '_' + em + '_itl_model_' +
                            'split' + str(fold) + '_CF')
            format_itl_cmd = ' '.join(['python ./utils/landmark_utils/Classification/FormatDataImageFolder.py',
                                       '--task '+ 'edgemapITLkfold',
                                       '--dataset '+ dataset,
                                       '--neu_img_folder ' + test_neu_folder,
                                       '--emo_lnd_folder ' + emo_lnd_folder,
                                       '--out_folder ' + format_itl_out,
                                       '--dirname_protocol ' + dirname_protocol])
            with subprocess.Popen(format_itl_cmd, shell=True, stdout=subprocess.PIPE) as cmd:
                for line in cmd.stdout:
                    print(line)

            # Do classification
            classify_itl_cmd = ' '.join(['python ./utils/landmark_utils/Classification/test_lnd.py',
                                         '--model_path ' + model_path,
                                         '--dataset ' + dataset,
                                         '--data_dir ' + format_itl_out])
            with subprocess.Popen(classify_itl_cmd, shell=True, stdout=subprocess.PIPE) as cmd:
                for line in cmd.stdout:
                    if "Test accuracy" in line.decode("utf-8"):
                        em_test_costs[em]['accuracy'].append(float(line.decode("utf-8").split(' ')[-1].strip('\n')))

elif itl_type == 'joint':
    for fold in range(1, num_fold + 1):
        print(fold)
        count = 0
        itl_cmd = ' '.join(['python itl_joint_argparse.py',
                            '--dataset ' + dataset,
                            '--inc_emotion',
                            '--kfold ' + str(fold),
                            '--save_pred',
                            '--save_model',
                            '--output_folder '+itl_out_folder])
        with subprocess.Popen(itl_cmd, shell=True, stdout=subprocess.PIPE) as cmd:
            for line in cmd.stdout:
                if "test cost" in line.decode("utf-8"):
                    line_dec = line.decode("utf-8")
                    list_dec = json.loads(line_dec[17:-2])
                    for i, em in enumerate(all_emotions):
                        em_test_costs[em]['cost'].append(list_dec[i])
                if "TC1Num" in line.decode("utf-8"):
                     print(line)
        # Format ITL to Edgemaps
        emo_lnd_folder = os.path.join(itl_out_folder, dataset + '_itl_model_' +
                        'split' + str(fold) + '_CF', 'predictions', dataset)
        format_itl_out = os.path.join(format_out_folder, dataset + '_itl_model_' +
                        'split' + str(fold) + '_CF')
        format_itl_cmd = ' '.join(['python ./utils/landmark_utils/Classification/FormatDataImageFolder.py',
                                   '--task '+ 'edgemapITLJoint',
                                   '--dataset '+ dataset,
                                   '--neu_img_folder ' + test_neu_folder,
                                   '--emo_lnd_folder ' + emo_lnd_folder,
                                   '--out_folder ' + format_itl_out,
                                   '--dirname_protocol ' + dirname_protocol])
        with subprocess.Popen(format_itl_cmd, shell=True, stdout=subprocess.PIPE) as cmd:
            for line in cmd.stdout:
                print(line)

        # Do classification
        classify_itl_cmd = ' '.join(['python ./utils/landmark_utils/Classification/test_lnd.py',
                                     '--model_path ' + model_path,
                                     '--dataset ' + dataset,
                                     '--data_dir ' + format_itl_out,
                                     '--joint'])
        with subprocess.Popen(classify_itl_cmd, shell=True, stdout=subprocess.PIPE) as cmd:
            for line in cmd.stdout:
                if "emotion wise" in line.decode("utf-8"):
                    count = 1
                if count == 1:
                    if line.decode("utf-8").split(' ')[0] in all_emotions:
                        em = line.decode("utf-8").split(' ')[0]
                        acc = float(line.decode("utf-8").split(' ')[-1].strip('\n'))
                        em_test_costs[em]['accuracy'].append(acc)
                if "Test accuracy" in line.decode("utf-8"):
                    print(line)

print(time.time()-start_time)

print(em_test_costs)

plot_box(em_test_costs, dataset, itl_type, itl_out_folder)



# def plot_error_bars(plot_dict):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     emos = plot_dict.keys()
#     x_pos = np.arange(len(emos))
#     cost_mean = [];
#     cost_dev = np.zeros((2, len(emos)))
#
#     for i, em in enumerate(emos):
#         cost_mean.append(np.mean(plot_dict[em]))
#         #cost_std.append(np.std(plot_dict[em]))
#         cost_dev[0, i] = np.abs((plot_dict[em] - np.mean(plot_dict[em])).min())
#         cost_dev[1, i] = np.abs((plot_dict[em] - np.mean(plot_dict[em])).max())
#
#     # Build the plot
#     fig, ax = plt.subplots()
#     ax.bar(x_pos, cost_mean, yerr=cost_dev, align='center', alpha=0.5, ecolor='black', capsize=10)
#     ax.set_ylabel('Reconstruction error')
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(emos)
#     ax.set_title('10-fold reconstruction error')
#     ax.yaxis.grid(True)
#
#     # Save the figure and show
#     plt.tight_layout()
#     # plt.savefig('bar_plot_with_error_bars.png')
#     plt.show()
