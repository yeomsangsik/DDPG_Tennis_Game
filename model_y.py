import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, random_seed):
        super(Actor, self).__init__()
        
        
        self.seed = torch.manual_seed(random_seed)
        self.linear1 = nn.Linear(state_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, action_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(*hidden_init(self.linear3))
        self.linear4.weight.data.uniform_(*hidden_init(self.linear4))
        self.linear5.weight.data.uniform_(-3e-3, 3e-3)     
    
    def forward(self, state):
        x= F.relu(self.linear1(state))
        x= F.relu(self.linear2(x))
        x= F.relu(self.linear3(x))
        x= F.relu(self.linear4(x))
        
        return torch.tanh(self.linear5(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, random_seed):
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(random_seed)
        self.linear1 = nn.Linear(state_size, 1024)
        self.linear2 = nn.Linear(1024 + action_size, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(*hidden_init(self.linear3))
        self.linear4.weight.data.uniform_(*hidden_init(self.linear4))
        self.linear5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):

        xs = F.relu(self.linear1(state))
        x  = torch.cat((xs, action), dim=1)
        x  = F.relu(self.linear2(x))
        x  = F.relu(self.linear3(x))
        x  = F.relu(self.linear4(x))
        
        
        return self.linear5(x)      
              
