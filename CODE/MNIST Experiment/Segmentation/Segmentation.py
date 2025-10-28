from Revised_Focal_loss import sigmoid_focal_loss_revised
from unet import UNet
import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import copy
import os
from Data_loader_upscale_mnist import DriveDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
Sigmoid=torch.nn.Sigmoid()

device = torch.device("cuda:1")
Patch_size=28
path_to_results=""
if os.path.isdir(path_to_results)==False:
    os.makedirs(path_to_results)

if Patch_size!=28:
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((Patch_size,Patch_size)),
    ])

else:
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ])

Path_to_MNIST='/home/mleeuwen/DATA/MNIST/train'
mnist_trainset=datasets.MNIST(root=Path_to_MNIST,train=True,download=False,transform=transforms)

batch_size=64
epochs=1000
gamma=0.5
alpha=0.5
th=3
easy=False

train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
Noise_levels=[0,0.5,0.75]
running_loss=0.0
columns = ['Noise']+[str(i) for i in range(0,epochs)]
All_training_losses=np.ones((len(Noise_levels),epochs+1))*-1
df_epoch = pd.DataFrame (All_training_losses)

Epochs_losses=[]
Experiment_nr=0

for nl in range(len(Noise_levels)):

    Unet=UNet(channels=1).to(device)
    optimizer = optim.SGD(Unet.parameters(), lr=0.001, momentum=0.9)

    noise_amplitude=Noise_levels[nl]
    All_training_losses[Experiment_nr,0]=noise_amplitude

    for epoch in range(epochs):

        running_loss=0.0
        batch=0

        for inputs, labels in train_loader:

                optimizer.zero_grad()

                noise=torch.tensor(np.random.rand(len(inputs),1,28, 28)*noise_amplitude)

                labels=copy.copy(inputs)>0.5
                labels=labels.type(torch.float)
                labels=labels.to(device)

                inputs=inputs+noise
                inputs=np.clip(inputs,0.0,1.0).type(torch.float)
                inputs=inputs.to(device)
                outputs = Unet(inputs)

                #loss = torchvision.ops.sigmoid_focal_loss(outputs, labels,alpha=alpha,gamma=gamma,reduction = 'mean')
                loss= sigmoid_focal_loss_revised(outputs, labels,alpha=alpha,gamma=gamma,reduction = 'mean',epsilon_scalar=1e-3)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        running_loss=running_loss/len(train_loader)
        Epochs_losses.append(running_loss)

        if epoch%100==0:
            sigoutput=Sigmoid(outputs)
            raw_output=outputs

            fig,ax=plt.subplots(1,4,figsize=(10,5),dpi=300)
            ax[3].imshow(inputs[0,0,:,:].detach().cpu()+sigoutput[0,0,:,:].detach().cpu(),cmap='gray',vmin=0,vmax=1)
            ax[2].imshow(sigoutput[0,0,:,:].detach().cpu(),cmap='gray',vmin=0,vmax=1)
            ax[1].imshow(labels[0,0,:,:].detach().cpu(),cmap='gray',vmin=0,vmax=1)

            ax[0].imshow(inputs[0,0,:,:].detach().cpu(),cmap='gray',vmin=0,vmax=1)
            plt.savefig(os.path.join(path_to_results,'Prediction_at_epoch%s_%s.png'%(epoch,nl)))
            print(running_loss,epoch)

            print("Max value = %s"%torch.max(raw_output[0,0,:,:].detach().cpu()))
            print("Min value = %s"%torch.min(raw_output[0,0,:,:].detach().cpu()))
            plt.close()

        if torch.isnan(loss)==True:
            print('Terminated due to NaN at epoch %s'%epoch)
            df_epoch.iloc[Experiment_nr,epoch+1]='inf'

            break
        else:
            df_epoch.iloc[Experiment_nr,epoch+1]=running_loss

    Experiment_nr+=1
    print('Finished Training')

df_epoch.iloc[:,:]=df_epoch.iloc[:,:].replace(-1,'inf')
df_epoch.columns=columns

filepath_epochs = os.path.join(path_to_results,
                'Experiment_Results_no_noise_new_loss_epoch.xlsx')
df_epoch.to_excel(filepath_epochs, index=False)
