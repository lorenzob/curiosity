from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

def set_seed(seed):
  print("SEED", seed)
  torch.manual_seed(seed)
  np.random.seed(seed)

set_seed(1234)


class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.layer = torch.nn.Linear(1, 1)

   def forward(self, x):
       x = self.layer(x)      
       return x

net = Net()
print(net)

#Net(
#  (hidden): Linear(in_features=1, out_features=1, bias=True)
#)

# Visualize our data
import matplotlib.pyplot as plt

x = np.random.rand(100)
y = np.sin(x) * np.power(x,3) + 3*x + np.random.rand(100)*0.8

plt.scatter(x, y)
#plt.show()

# convert numpy array to tensor in shape of input size
x = torch.from_numpy(x.reshape(-1,1)).float()
y = torch.from_numpy(y.reshape(-1,1)).float()
#print(x, y)

# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

val_x = np.random.rand(100)
val_y = np.sin(val_x) * np.power(val_x,3) + 3*val_x
val_x = torch.from_numpy(val_x.reshape(-1,1)).float()
val_y = torch.from_numpy(val_y.reshape(-1,1)).float()

val_inputs = Variable(val_x)
val_outputs = Variable(val_y)

inputs = Variable(x)
outputs = Variable(y)
for i in range(500):
   prediction = net(inputs)
   loss = loss_func(prediction, outputs) 

   # save individual losses for curiosity training
   all_losses = F.mse_loss(prediction, outputs, reduction="none")

   optimizer.zero_grad()
   loss.backward()        
   optimizer.step()       

   # extra training on worst elements
   curiosity = False
   if curiosity:

     # select worst items
     k = 20
     all_losses = all_losses.data.numpy().reshape(100,)
     worst = np.argpartition(all_losses, -k)
     worst_idx = worst[-k:]

     r_inputs = inputs[worst_idx]
     r_outputs = outputs[worst_idx]

     # train
     r_prediction = net(r_inputs)
     r_loss = loss_func(r_prediction, r_outputs)
     optimizer.zero_grad()
     r_loss.backward()        
     optimizer.step()       

   if i % 10 == 0:
       # plot and show learning process
       val_prediction = net(val_inputs)
       val_loss = loss_func(val_prediction, val_outputs)

       plt.cla()
       plt.scatter(val_x.data.numpy(), val_y.data.numpy())
       plt.plot(val_x.data.numpy(), val_prediction.data.numpy(), 'r-', lw=2)
       plt.text(0.5, 0, 'Loss=%.4f' % val_loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
       plt.pause(0.1)

       print("loss", val_loss.data.numpy())

plt.show()