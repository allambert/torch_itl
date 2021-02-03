import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse

config = argparse.ArgumentParser()
config.add_argument("--dataset", type=str, help="name of the dataset KDEF or Rafd")
config.add_argument("--data_dir", type=str, help="name of the emo_lnd_folder")
config.add_argument("--model_path", type=str, help="name of the out_folder")
config.add_argument("--joint", action="store_true",
                    help="whether to compute for joint")
args = config.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#MODEL_PATH = args.model_path
#MODEL_PATH = './LndExperiments/KDEF_bs16_e10_20201117-181507'
# MODEL_PATH  = './LndExperiments/Rafd_bs16_e10_20201118-055249'
MODEL_PATH = './LndExperiments/RafdwoCON_bs16_e10_20201203-133925'
#MODEL_PATH  = './LndExperiments/Aff_bs16_e10_20210103-213431'

dataset = 'KDEF' #args.dataset #'KDEF'
joint_compute = False #args.joint
batch_size = 2
input_size = 224

if dataset == 'KDEF':
    num_classes = 7
    #data_dir = args.data_dir  # './EM_Classification_Exp/LndPredJointInpEmoWise/NE'
    # data_dir = './LndPredKDEF_itl_model_20201118-134545'
    #data_dir = './KDEF_LandmarkClassification_rafd/train'
    #data_dir = './KDEF_LandmarkClassification_rafd/test'
    data_dir = './EM_Classification_Exp/starGAN_50kdec25k'
elif dataset == 'Rafd':
    num_classes = 7 #making wo CON
    data_dir = args.data_dir #'./LndPredRafd_itl_model_20201118-134437'
    # data_dir = './Rafd_LandmarkClassification/train'
    data_dir = './RafdwoCON_LandmarkClassification/test'
    # data_dir = './RafdwoCON_LandmarkClassification_kdef/test'
elif dataset == 'RafdwoCON':
    num_classes = 7
    # data_dir = './LndPredRafd_neutral_itl_model_20201203-180453'
    # data_dir = './RafdwoCON_LandmarkClassification/train'
    # data_dir = './RafdwoCON_LandmarkClassification/test'

# dataloader and transform
data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])

print(dataset, data_dir, MODEL_PATH, joint_compute)


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


#image_dataset = datasets.ImageFolder(data_dir, data_transform)
image_dataset = ImageFolderWithPaths(data_dir, data_transform)
test_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(image_dataset.class_to_idx)


# model def
def model(model_name, num_classes):
    if model_name == 'resnet-18':
        model_ft = models.resnet18(pretrained=False)
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


# Get ResNet and load wts
emo_model_ft = model('resnet-18', num_classes)
emo_model_ft.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
emo_model_ft = emo_model_ft.to(device)
emo_model_ft.eval()


correct = 0
total = 0
y_true_list = []
y_pred_list = []

in_out_data = []
with torch.no_grad():
    for data in test_loader:
        inputs, labels, paths = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = emo_model_ft(inputs)
        _, predicted = torch.max(outputs.data, 1)
        if 'neutral' in paths[0]:
            print(paths[0], outputs[0])
        # print(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true_list.append(labels.numpy())
        y_pred_list.append(predicted.numpy())
        if joint_compute:
            if dataset == 'KDEF':
                for j, im_path in enumerate(paths):
                    in_out_data.append({'out_em': im_path.split('/')[-2],
                                        'in_em': im_path.split('/')[-1][9:11],
                                        'pred': 1*(predicted.numpy()[j] == labels.numpy()[j])
                                        })
            elif dataset == 'Rafd':
                for j, im_path in enumerate(paths):
                    in_out_data.append({'out_em': im_path.split('/')[-2],
                                        'in_em': im_path.split('/')[-1].split('_')[5],
                                        'pred': 1 * (predicted.numpy()[j] == labels.numpy()[j])
                                        })
print('Test accuracy:', 100 * correct / total)

y_true = np.concatenate(y_true_list)
y_pred = np.concatenate(y_pred_list)
print(confusion_matrix(y_true, y_pred))

if joint_compute:
    import pandas as pd
    in_out_data = pd.DataFrame.from_records(in_out_data)
    all_em = in_out_data['in_em'].unique().tolist()
    print('emotion wise')
    for em in all_em:
        calc = in_out_data[in_out_data['in_em'] == em]['pred'].tolist()
        print(em, sum(calc)/len(calc)*100)
