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

size = 10000
x = np.random.rand(size)
y = np.sin(x) * np.power(x,3) + 3*x + np.random.rand(size)*0.8

# convert numpy array to tensor in shape of input size
x = torch.from_numpy(x.reshape(-1,1)).float()
y = torch.from_numpy(y.reshape(-1,1)).float()

inputs = Variable(x)
outputs = Variable(y)
for i in range(250):

   BATCH_SIZE = len(inputs)

   prediction = net(inputs)
   loss = loss_func(prediction, outputs) 

   # save individual losses for curiosity training
   all_losses = F.mse_loss(prediction, outputs, reduction="none")

   optimizer.zero_grad()
   loss.backward()        
   optimizer.step()

   # extra training on worst elements
   curiosity = True
   curiosity_ratio = 0.2
   k = int(round(BATCH_SIZE * curiosity_ratio))
   if curiosity:
     # select worst items
     all_losses = all_losses.data.numpy().reshape(size,)
     worst = np.argpartition(all_losses, -k)
     retry_idx = worst[-k:]
   else:
     indexes = np.arange(BATCH_SIZE)
     retry_idx = np.random.choice(indexes, size=k, replace=False)

   r_inputs = inputs[retry_idx]
   r_outputs = outputs[retry_idx]

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