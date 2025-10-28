import torch
import torchvision
import numpy as np
import sigmoid_focal_loss_modified
Sigmoid=torch.nn.Sigmoid()

alpha=0.5
gamma=2
steps=10000
outputs=np.linspace(-38,38,steps)

labels=torch.tensor(1.,dtype=torch.float64) # GT label, 0.0 for background gradient
y=Sigmoid(torch.tensor(outputs,dtype=torch.float64))
Stable_loss=False  # switch to True is you want to use the stabilized loss
epsilon= 1e-3

for i in range(steps):
    y_t=torch.tensor(outputs[i]*1.0,requires_grad=True)
    if Stable_loss==False:
        loss=torchvision.ops.sigmoid_focal_loss(y_t,labels,
             alpha=alpha,gamma=gamma,reduction='mean)
    else:
        loss=sigmoid_focal_loss_modified(y_t,labels,
             alpha=alpha,gamma=gamma,reduction='mean,
             epsilon=epsilon)
         
    loss.backward()
    Gradient=y_t.grad.item()
    print(Gradient)
