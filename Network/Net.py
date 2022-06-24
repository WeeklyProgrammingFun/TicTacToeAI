#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ToeDataset(Dataset):
    def __init__(self,filename):
        # list of 9 tuples, each a board state: 0=blank,1=X, -1=O
        #samples = [] 
        # list of good replies, 0 = bad, 1 = good
        #truth   = [] 
        
        self.data = torch.from_numpy(
            np.genfromtxt(filename, delimiter=",",dtype=np.float32)
            ) 
        #print(self.data)#.shape())
        #print(self.data.size()[0])
    def __len__(self):
        return self.data.size()[0]
    def __getitem__(self,index):
        d = self.data[index]
        sample = {'state' : d[0:9], 'moves': d[9:18] }
        return sample

fn = 'TicTacToeData.txt'
ds = ToeDataset(fn)

class AI_1(nn.Module):
    def __init__(self):
        super(AI_1, self).__init__()
        
        # 2 corners, 350 size internals works
        # 2 corners, 275 size internals fails
        # MSE size 350, 500 epoch works
        
        
        board_size = 9
        internal1  = 350 # big enough to memorize 4500ish entries
        self.net   = nn.Sequential(
            nn.Linear(board_size, internal1),
            
            nn.ReLU(),
            
            nn.Linear(internal1, internal1),
            
            #nn.ReLU(),
            
            #nn.Linear(internal1, internal1),
            
            #nn.ReLU(),

            #nn.Linear(internal1, internal1),
            
            nn.ReLU(),            

            nn.Linear(internal1, board_size)                      
            );
    def forward(self, in_data):
        x = self.net(in_data)
        return x


net1 = AI_1()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Get the cuda device
print(device)
net1.to(device)

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def train():
    writer = SummaryWriter(log_dir=None, comment="TicTacToeTestAdam")
    net = AI_1().to(device)
    fn = 'TicTacToeData.txt'
    dataset = ToeDataset(fn)
    loader = DataLoader(dataset, shuffle=True, batch_size = 300)
    
    loss_func = nn.MSELoss() # 0.11 in 500 epochs
    #loss_func = nn.L1Loss()  # still has 220 errors at 1500 epochs, 
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    #optimizer = torch.optim.RMSprop(net.parameters(),lr=0.01)
    #optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
    
    max_epoch = 500
    for epoch in range(max_epoch):
        dataiter = iter(loader)
        
        epoch_loss = 0
        
        for batch in dataiter:
            optimizer.zero_grad()
            
            out = net(batch['state'].to(device))
            
            loss = loss_func(out, batch['moves'].to(device))
            epoch_loss += loss.item()
            
            loss.backward()            
            
            optimizer.step()
            
        print(f'epoch/max {epoch}/{max_epoch} loss {epoch_loss}')
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        
        
    sample = dataset[100]
    #print(sample)
    out1 = net(sample['state'].to(device))
    #print(out1)
    writer.add_graph(net, batch['state'].to(device))
    writer.flush()
    writer.close()
    return net


net = train()

test1 = torch.FloatTensor([1,0,0,0,0,0,0,0,0]).to(device)
net.forward(test1)

def computeOne(index):
    test1 = torch.FloatTensor(ds[index]['state']).to(device)
    return net.forward(test1)
    
def drawOne(index):
    test1 = torch.FloatTensor(ds[index]['state']).to(device)
    print(net.forward(test1) - torch.FloatTensor(ds[index]['moves']).to(device))

def cleaner(index):
    # turn vector into 0,1 best guesses
    vec = computeOne(index)
    #vec = torch.FloatTensor([1,0,0,0,0,0,0,0,0]).to(device)
    #vec = net.forward(vec)
    max = torch.max(vec).item()
    min = torch.min(vec).item()
    # cleaned = (vec-min)/(max-min) > 0.5
    cleaned = vec > 0.5
    cc = cleaned.float()
    return cc
    # print(max,min,vec,cleaned,cc)

# check all match truth
# todo - rewrite to do all in one pass on GPU
def checkAll():
    failed = 0
    for i in range(len(ds)):
        ans = cleaner(i)
        truth = ds[i]['moves'].to(device)
        diff = torch.max(abs(ans-truth)).item()
        if diff > 0:
            #print(i,ans,truth,diff)
            failed = failed + 1
    print('failed:',failed)

checkAll()