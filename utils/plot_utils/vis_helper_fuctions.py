import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

base_output_path = './visualizations'


# func for generating cost/accuracy plot tables from output dictionaries for single and joint
def plot_box(plot_dict): 
    import pandas as pd 
    import matplotlib.pyplot as plt 
 
    for subkey in ['cost', 'accuracy']: 
        sub_dict = {key: plot_dict[key][subkey] for key in plot_dict.keys()} 
        df = pd.DataFrame.from_dict(sub_dict) 
        print(df) 
        if subkey == 'cost': 
            print(pd.concat([df.mean().round(3),df.std().round(3)], axis=1).to_latex()) 
            print(df.mean(axis=1).mean(),df.mean(axis=1).std()) 
        else: 
            print(pd.concat([df.mean().round(2),df.std().round(2)], axis=1).to_latex()) 
            print(df.mean(axis=1).mean(), df.mean(axis=1).std()) 


# func generating image grids for various qualitative results and illutrations
def plot_grid_1D(image_list, label_list, row_labels, out_img_name, fig_title=None, cmap=None):
    rows = len(image_list)
    columns = len(image_list[0])
    assert len(image_list[0]) == len(label_list)

    fig = plt.figure(figsize=(2*columns, 2*rows))
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)
    for i in range(rows):
        for j in range(columns):
            ax = fig.add_subplot(rows, columns, (i*columns) + j + 1)
            img = mpimg.imread(image_list[i][j])
            if cmap == None:
                imgplot = plt.imshow(img)
            else:
                imgplot = plt.imshow(img, cmap=cmap)
            #plt.axis('off')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if i == 0:
                ax.set_title(label_list[j])
            if j == 0:
                if not row_labels ==[]:
                    print(row_labels[i])
                    ax.set_ylabel(row_labels[i], rotation=90, size='large')
    
    fig.subplots_adjust(wspace=0.05)
    plt.savefig(os.path.join(base_output_path, out_img_name), bbox_inches='tight', pad_inches=0.01)

# emotion va space plot generation
def plot_emotion_emb(emo_va, out_img_name):
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    circ = plt.Circle((0, 0), radius=1., edgecolor='k', facecolor='None')
    ax.add_patch(circ)
    for key in emo_va.keys():
        if not key == 'Neutral':
            eva = emo_va[key] / np.linalg.norm(emo_va[key])
            ax.scatter(eva[0], eva[1])
            ax.annotate(key, (eva[0], eva[1]))
        else:
            ax.scatter(0, 0)
            ax.annotate(key, (0, 0))
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    plt.grid(True)
    plt.savefig(os.path.join(base_output_path, out_img_name), bbox_inches='tight', pad_inches=0.1)

# helper script to know how utility functions can be utilized
if __name__ == "__main__":
    task = 'kdef_edgemap_illustration'


    if task == 'kdef_edgemap_illustration':
        base_path = 'add_data_base_path'
        imlist = [[os.path.join(base_path, 'NE/AF02NES.JPG'),
                   os.path.join(base_path, 'AF/AF02AFS.JPG'),
                   os.path.join(base_path, 'AN/AF02ANS.JPG'),
                   os.path.join(base_path, 'DI/AF02DIS.JPG'),
                   os.path.join(base_path, 'HA/AF02HAS.JPG'),
                   os.path.join(base_path, 'SA/AF02SAS.JPG'),
                   os.path.join(base_path, 'SU/AF02SUS.JPG'),
                   ]]
        label_list = ['Neutral', 'Fearful', 'Angry', 'Disgusted', 'Happy', 'Sad', 'Surprised']
        out_im_name = 'kdef_edgemap_iilustration.pdf'
        plot_grid_1D(imlist, label_list, [], out_im_name, cmap='gray')


    elif task == 'rafd_edgemap_illustration':
        gt_base_path = 'add_data_base_path_here'
        imlist = [[os.path.join(gt_base_path, 'neutral/Rafd090_01_Caucasian_female_neutral_frontal.JPG'),
                   os.path.join(gt_base_path, 'fearful/Rafd090_01_Caucasian_female_fearful_frontal.JPG'),
                   os.path.join(gt_base_path, 'angry/Rafd090_01_Caucasian_female_angry_frontal.JPG'),
                   os.path.join(gt_base_path, 'disgusted/Rafd090_01_Caucasian_female_disgusted_frontal.JPG'),
                   os.path.join(gt_base_path, 'happy/Rafd090_01_Caucasian_female_happy_frontal.JPG'),
                   os.path.join(gt_base_path, 'sad/Rafd090_01_Caucasian_female_sad_frontal.JPG'),
                   os.path.join(gt_base_path, 'surprised/Rafd090_01_Caucasian_female_surprised_frontal.JPG'),
                   ]]
        label_list = ['Neutral', 'Fearful', 'Angry', 'Disgusted', 'Happy', 'Sad', 'Surprised']
        out_im_name = 'rafd_edgemap_iilustration.pdf'
        plot_grid_1D(imlist, label_list, [], out_im_name, cmap='gray')


    elif task == 'emo_va_plot':
        from datasets import import_affectnet_va_embedding
        emo_va = import_affectnet_va_embedding('')
        plot_emotion_emb(emo_va, 'emo_va_vis.eps')


    elif task == 'kdef_discrete_plots':
        row_labels = ['Ground Truth', 'vITL', 'StarGAN']
        column_labels = ['Neutral', 'Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised']
        
        gt_base_path = 'gt_path'
        star_base_path = 'stargan_path'
        itl_base_path = 'itl_split_path'

        gt_list = [os.path.join(gt_base_path, 'NE/AF07NES.JPG'),
                   os.path.join(gt_base_path, 'AN/AF07ANS.JPG'),
                   os.path.join(gt_base_path, 'DI/AF07DIS.JPG'),
                   os.path.join(gt_base_path, 'AF/AF07AFS.JPG'),
                   os.path.join(gt_base_path, 'HA/AF07HAS.JPG'),
                   os.path.join(gt_base_path, 'SA/AF07SAS.JPG'),
                   os.path.join(gt_base_path, 'SU/AF07SUS.JPG'),
                   ]
        star_list =[os.path.join(star_base_path, 'neutral/pred_AF07NES.JPG'),
                   os.path.join(star_base_path, 'angry/pred_AF07NES.JPG'),
                   os.path.join(star_base_path, 'disgusted/pred_AF07NES.JPG'),
                   os.path.join(star_base_path, 'fearful/pred_AF07NES.JPG'),
                   os.path.join(star_base_path, 'happy/pred_AF07NES.JPG'),
                   os.path.join(star_base_path, 'sad/pred_AF07NES.JPG'),
                   os.path.join(star_base_path, 'surprised/pred_AF07NES.JPG'),
                   ]
        itl_list = [os.path.join(itl_base_path, 'neutral/pred_AF07NES.JPG'),
                   os.path.join(itl_base_path, 'angry/pred_AF07NES.JPG'),
                   os.path.join(itl_base_path, 'disgusted/pred_AF07NES.JPG'),
                   os.path.join(itl_base_path, 'fearful/pred_AF07NES.JPG'),
                   os.path.join(itl_base_path, 'happy/pred_AF07NES.JPG'),
                   os.path.join(itl_base_path, 'sad/pred_AF07NES.JPG'),
                   os.path.join(itl_base_path, 'surprised/pred_AF07NES.JPG'),
                   ]

        imlist = [gt_list, itl_list, star_list]
        out_im_name = 'kdef_discrete_plots.pdf'
        plot_grid_1D(imlist, column_labels, row_labels, out_im_name, cmap='gray')


    elif task == 'kdef_confusion':
        cm_kdef = np.array([[93,  5,  0,  0,  0,  0,  0],
                            [ 2, 96,  0,  0,  0,  0,  0],
                            [ 0, 16, 32,  0, 25, 25,  0],
                            [ 0,  0,  0, 98,  0,  0,  0],
 			    [ 3, 0,  0,  0, 76, 19,  0],
 		            [29,  1,  0,  0, 22, 46,  0],
 			    [ 0,  0, 20,  0,  0,  0, 78]])
        classes = ['Angry','Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        cm_kdef = np.round(cm_kdef/98., 2)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_kdef,
                               display_labels=classes)
        disp.plot(xticks_rotation=30., ax=ax)
        disp.im_.colorbar.remove()
        # plt.rcParams['xtick.labelsize']=25
        # plt.rcParams['ytick.labelsize']=25
        # plt.rcParams['font.size'] = 20
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.savefig(os.path.join(base_output_path, 'kdef_confusion.pdf'), bbox_inches='tight', pad_inches=0.01)
    
    elif task == 'rafd_confusion':
        
        cm_rafd = np.array([[ 22, 0,  0,  0,  0, 20,  0],
 			    [ 0,  41, 1,  0,  0,  0,  0],
                            [ 0,  0,  28, 0 , 0,  0, 14],
 			    [ 0,  0,  0, 42,  0,  0,  0],
 			    [ 0,  0,  0,  0, 17, 25,  0],
 			    [ 0,  0,  0,  0,  2, 40,  0],
 			    [ 0,  0,  0,  0,  0,  0, 42]])
        classes = ['Angry', 'Disgusted','Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        cm_rafd = np.round(cm_rafd/42., 2)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_rafd,
                               display_labels=classes)
        disp.plot(xticks_rotation=30., ax=ax)
        # plt.rcParams['xtick.labelsize']=25
        # plt.rcParams['ytick.labelsize']=25
        # plt.rcParams['font.size'] = 20
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.savefig(os.path.join(base_output_path, 'rafd_confusion.pdf'), bbox_inches='tight', pad_inches=0.01)
