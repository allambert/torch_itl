import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = './Experiments/KDEF_bs16_e15_20201116-071653'
dataset = 'KDEF'
batch_size = 2
input_size = 224

if dataset == 'KDEF':
    num_classes = 6
    data_dir = './SynthPredKDEF_itl_model_20201116-165010'
    # data_dir = './SynthGT_NONE'
    # data_dir = './KDEF_Classification_NONE/test'
elif dataset == 'Rafd':
    num_classes = 8
    data_dir = ''

# dataloader and transform
data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
image_dataset = datasets.ImageFolder(data_dir, data_transform)
test_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(image_dataset.class_to_idx)


# model def
def model(model_name, num_classes):
    if model_name == 'resnet-18':
        model_ft = models.resnet18(pretrained=False)
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

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = emo_model_ft(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true_list.append(labels.numpy())
        y_pred_list.append(predicted.numpy())
print('Test accuracy:', 100 * correct / total)

y_true = np.concatenate(y_true_list)
y_pred = np.concatenate(y_pred_list)
print(confusion_matrix(y_true, y_pred))