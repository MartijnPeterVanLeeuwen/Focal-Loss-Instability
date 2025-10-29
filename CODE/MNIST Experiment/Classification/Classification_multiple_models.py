import torch
import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(parent_dir)

from Stabilized_Focal_loss import sigmoid_focal_loss_modified
from Models.CNN import *

Path_to_MNIST_train='.//MNIST//DATA'
Path_to_Main_results='.//MNIST//RES'

mnist_trainset=datasets.MNIST(root=Path_to_MNIST_train,
                train=True,download=False,transform=torchvision.transforms.ToTensor())#Set "download" to True if you want to download the MNIST dataset

epsilon=1e-3
batch_size=64
epochs=100
gamma=0.5
alpha=0.5
lr=1e-3
Add_noise=True
Device='cpu' #Specify the GPU or CPU

columns = ['Gamma', 'TH']+[str(i) for i in range(0,epochs)]
gamma_values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0,1,2,3,4,5]
THs=[0,1,2,3,4,5,6,7,8]
noise_amplitudes=[0,0.5,0.75]
Loss_functions=['Adapted','Original']

Sigmoid=torch.nn.Sigmoid()
device = torch.device(Device)

for Lf in Loss_functions:

    Path_to_experiments=os.path.join(Path_to_Main_results,"Focal_loss_%s"%Lf)

    if os.path.isdir(Path_to_experiments)==False:
        os.makedirs(Path_to_experiments)

    for Na in noise_amplitudes:

        result_folder=os.path.join(Path_to_experiments,"Noise_level_%s"%Na)

        if os.path.isdir(result_folder)==False:

            All_training_losses=np.ones((len(gamma_values)*len(THs),epochs+2))*-1
            All_training_acc=np.ones((len(gamma_values)*len(THs),epochs+2))*-1

            df_epoch = pd.DataFrame (All_training_losses)
            df_ac = pd.DataFrame (All_training_acc)

            os.mkdir(result_folder)
            noise_amplitude=Na
            random.seed(123)

            Experiment_nr=0
            for GAMMA in range(len(gamma_values)):
                for TH in range(len(THs)):

                    All_training_losses[Experiment_nr,0]=gamma_values[GAMMA]
                    All_training_losses[Experiment_nr,1]=THs[TH]
                    All_training_acc[Experiment_nr,0]=gamma_values[GAMMA]
                    All_training_acc[Experiment_nr,1]=THs[TH]

                    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size,
                                                              shuffle=True)
                    dataiter = iter(trainloader)
                    images, labels = next(dataiter)

                    net=CNN(1,28,no_classes=1).to(device)
                    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
                    Train_loss=np.ones((1,epochs))*-1
                    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
                        running_loss = 0.0
                        train_acc=0
                        for i, data in enumerate(trainloader, 0):
                            # get the inputs; data is a list of [inputs, labels]
                            inputs, labels = data
                            for ii in range(len(labels)):
                                if labels[ii]>THs[TH]:
                                    labels[ii]=torch.tensor(0.,dtype=torch.float64)

                                    if Add_noise:
                                        noise=np.random.rand(28, 28)*noise_amplitude
                                        inputs[ii]=inputs[ii]+noise
                                        inputs[ii]=np.clip(inputs[ii],0,1)
                                else:
                                    labels[ii]=torch.tensor(1.)
                                    if Add_noise:
                                        noise=np.random.rand(28, 28)*noise_amplitude
                                        inputs[ii]=inputs[ii]+noise
                                        inputs[ii]=np.clip(inputs[ii],0,1)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            inputs=inputs.to(device)

                            # forward + backward + optimize
                            outputs = net(inputs)
                            labels=torch.unsqueeze(labels,1).float()
                            labels=labels.to(device)

                            if Lf=="Original":
                                loss = torchvision.ops.sigmoid_focal_loss(outputs, labels,alpha=alpha,gamma=gamma_values[GAMMA],reduction = 'mean')
                            else:
                                loss = sigmoid_focal_loss_modified(outputs, labels,alpha=alpha,gamma=gamma_values[GAMMA],reduction = 'mean',epsilon=epsilon)

                            loss.backward()
                            optimizer.step()

                            running_loss += loss.item()

                            outputs=Sigmoid(outputs)
                            outputs=torch.round(outputs)

                            train_acc += torch.sum(outputs == labels).item()

                        Epoch_accuracy=train_acc/len(mnist_trainset)

                        Epoch_loss= running_loss/len(trainloader)

                        if torch.isnan(loss)==True:
                            print('Terminated due to NaN at epoch %s'%epoch)
                            df_epoch.iloc[Experiment_nr,epoch+2]='inf'
                            df_ac.iloc[Experiment_nr,epoch+2]='inf'
                            break
                        else:
                            df_epoch.iloc[Experiment_nr,epoch+2]=Epoch_loss
                            df_ac.iloc[Experiment_nr,epoch+2]=Epoch_accuracy


                    print('Finished Training')

                    df_epoch.iloc[Experiment_nr,:]=df_epoch.iloc[Experiment_nr,:].replace(-1,'inf')
                    df_ac.iloc[Experiment_nr,:]=df_ac.iloc[Experiment_nr,:].replace(-1,'inf')

                    df_epoch.columns=columns
                    df_ac.columns=columns

                    filepath_epochs = os.path.join(result_folder,
                                    'Experiment_Results_loss.xlsx')
                    filepath_accuracy = os.path.join(result_folder,
                                    'Experiment_Results_acc.xlsx')

                    df_epoch.to_excel(filepath_epochs, index=False)
                    df_ac.to_excel(filepath_accuracy, index=False)

                    Experiment_nr+=1
