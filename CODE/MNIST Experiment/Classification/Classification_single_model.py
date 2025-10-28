import torch
import torchvision
import torchvision.datasets as datasets
from Simple_CNN import Net
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0")

mnist_trainset=datasets.MNIST(root='/home/mleeuwen/DATA/MNIST/train',train=True,download=False,transform=torchvision.transforms.ToTensor())
mnist_testset=datasets.MNIST(root='/home/mleeuwen/DATA/MNIST/test',train=False,download=False,transform=torchvision.transforms.ToTensor())

batch_size=64
epochs=100
gamma=0.5
alpha=0.5
th=3
easy=False
trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
dataiter = iter(trainloader)
images, labels = next(dataiter)

net=Net().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print('start_training')
Train_loss=[]

for epoch in range(epochs):  # loop over the dataset multiple times
    class_0=0
    class_1=0
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        for ii in range(len(labels)):
            if labels[ii]>th:
                labels[ii]=torch.tensor(0.,dtype=torch.float64)
                if easy==True:
                    inputs[ii]=torch.zeros(inputs[ii].shape)
                class_0+=1
            else:
                labels[ii]=torch.tensor(1.)
                class_1+=1

        # zero the parameter gradients
        optimizer.zero_grad()

        inputs=inputs.to(device)

        # forward + backward + optimize
        outputs = net(inputs)
        labels=torch.unsqueeze(labels,1).float()

        loss = torchvision.ops.sigmoid_focal_loss(outputs, labels,alpha=alpha,gamma=gamma,reduction = 'mean')
        loss.backward()
        optimizer.step()

        # print statistics
        Train_loss.append(loss.item())
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    if torch.isnan(loss)==True:
    	raise ValueError:
        	# Code that runs if that specific error occurs
        	print("You can't divide by zero!")



print('Finished Training')

