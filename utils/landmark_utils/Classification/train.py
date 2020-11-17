import os
import time
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = 'KDEF'
num_epochs = 1
batch_size = 4
input_size = 224

if dataset == 'KDEF':
    num_classes = 7
    data_dir = '/home/mlpboon/DatasetCheck'
elif dataset == 'Rafd':
    num_classes = 8
    data_dir = ''

# dataloader and transform
data_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=input_size, scale=(0.9, 1.0), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
image_dataset = datasets.ImageFolder(data_dir, data_transform)
train_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print(image_dataset.class_to_idx)


# model def
def model(model_name, num_classes):
    if model_name == 'resnet-18':
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


# Get ResNet
emo_model_ft = model('resnet-18', num_classes)
emo_model_ft = emo_model_ft.to(device)
emo_model_ft.train()
params_to_update = emo_model_ft.parameters()
# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        print(labels)
        optimizer_ft.zero_grad()
        outputs = emo_model_ft(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

# Save model
timestr = time.strftime("%Y%m%d-%H%M%S")
MODEL_PATH = os.path.join('./Experiments/', '_'.join((dataset,
                                                      'bs'+str(batch_size),
                                                      'e'+str(num_epochs), timestr)))
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.mkdir(os.path.dirname(MODEL_PATH))
torch.save(emo_model_ft.state_dict(), MODEL_PATH)