import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
import time
import torchvision
import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from Revised_Focal_loss import *
from Simple_CNN import *

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None,foreground=[0,1,8,9],background=[2,3,4,5,6,7]):
        self.img_dir = img_dir
        self.image_files=os.listdir(img_dir)
        self.transform = transform
        self.foreground=foreground
        self.background=background
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        im = Image.open(img_path)
        label = int(self.image_files[idx].split("_")[-1][:-4])
        if label in self.foreground:
            label=np.array([1.]).astype(np.float64)
        else:
            label=np.array([0.]).astype(np.float64)

        if self.transform:
            image = self.transform(im)

        return image, label


# Parameters
train_dir = ''
result_dir=''

if os.path.isdir(result_dir)==False:
    os.mkdir(result_dir)

batch_size = 128
num_workers = 8
num_epochs = 1000

device = torch.device("cuda:3")
model=Net(channels=3,output_classes=10).to(device)

# Transforms (ViT expects 224x224 input)
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


# Data Loaders
train_dataset=CustomImageDataset(train_dir, transform=train_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# Create ViT model
#model = create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
model.to(device)

# Loss, Optimizer, Scheduler
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

losses=np.ones(num_epochs)*-1.
accuracy=np.ones(num_epochs)*-1
Lf='Original'

print('start training')
# Training Loop
for epoch in range(num_epochs):
    print('epoch %s'%epoch)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        if Lf=="Original":
            loss = torchvision.ops.sigmoid_focal_loss(outputs, labels,alpha=0.5,gamma=0.5,reduction = 'mean')
        else:
            loss = sigmoid_focal_loss_revised(outputs, labels,alpha=0.5,gamma=0.5,reduction = 'mean',epsilon_scalar=1e-3)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        preds=torch.tensor(outputs>0.5,dtype=torch.uint8)
        labs=torch.tensor(labels,dtype=torch.uint8)

        correct += torch.sum(preds == labs).item()
        total += labs.size(0)

    train_acc = correct / total
    train_loss = running_loss / total

    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Time: {time.time() - start_time:.1f}s")

    if torch.isnan(loss)==True:
        print('terminated due to NaN')
        train_loss='NaN'

    losses[epoch]=train_loss
    accuracy[epoch]=train_acc

    Dictionary={"Losses":losses,"Accuracy":accuracy}
    df = pd.DataFrame.from_dict(Dictionary)
    df.to_excel(os.path.join(result_dir,'results.xlsx'))

    if torch.isnan(loss)==True:
        print('terminated due to NaN')
        break

# Save Model
torch.save(model.state_dict(), os.path.join(result_dir,'vit_base_patch16_224_imagenet.pth'))


