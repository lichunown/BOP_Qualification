# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:04:41 2017

@author: lichunyang
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.optim as optim

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,1)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = CNNNet()
 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2): 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): 
        inputs, labels = data   #data[4x3x32x32] tonsor length:4   
        inputs, labels = Variable(inputs), Variable(labels) 
        # zero the parameter gradients
        optimizer.zero_grad() 
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) #cross_entry
        loss.backward() 
        optimizer.step() #refresh
         
        # 每2000批数据打印一次平均loss值
        running_loss += loss.data[0]  #loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        if i % 2000 == 1999: # 每2000批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0