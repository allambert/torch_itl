from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import ToTensor, CenterCrop, Compose, Resize

from facenet_pytorch import InceptionResnetV1


# simple load image func
def load_img(file_path):
    img = Image.open(file_path)
    return img


# standardization as in facenet-pytorch
def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


if __name__ == "__main__":

    dataset = 'KDEF'
    dir_path = '../../../datasets/'
    use_csv = False

    inp_folder = dir_path + dataset + '_Aligned/' + dataset
    out_folder = dir_path + dataset + '_Aligned/' + dataset + '_facenet'

    if use_csv:
        # input data list
        df = pd.read_csv(inp_folder + '/' + dataset + '.csv')
        img_list = df['file_path'].tolist()
    else:
        img_list = []
        for dir, subdir, filenames in os.walk(inp_folder):
            for f in filenames:
                if f.endswith('.JPG'):
                    img_list.append(os.path.join(dir, f))
    print('files_to process', len(img_list))

    # img_list = ['../../../datasets/KDEF_Aligned/KDEF/AF01/AF01NES.JPG',
    #             '../../../datasets/KDEF_Aligned/KDEF/AF01/AF01ANS.JPG',
    #             '../../../datasets/KDEF_Aligned/KDEF/AF02/AF02NES.JPG']

    # create output folder
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # define transform on images
    transform = Compose([ToTensor()])

    # define network object
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    for im in img_list:
        # load PIL image
        img = load_img(im)
        # transform to tensor
        trans_img1 = transform(img)
        # reverse 255 div and standardize according to facenet-pytorch
        trans_img = fixed_image_standardization(255.0*trans_img1)
        # pass through net
        with torch.no_grad():
            img_embedding = resnet(trans_img.unsqueeze(0))
        # save embedding
        np.save(os.path.join(out_folder, os.path.basename(im).split('.')[0] + '.npy'),
                img_embedding.detach().numpy())
