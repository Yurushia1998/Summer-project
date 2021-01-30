import torch
from torch import nn
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        ip_emb = 42560
        emb1 = 1024
        emb2 = 256
        emb3 = 64
        emb4 = 16
        out_emb = 1
        
        self.layer1 = nn.Sequential(
        nn.Linear(ip_emb, emb1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3))

        
        self.layer2 = nn.Sequential(
        nn.Linear(emb1, emb2),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(emb2, emb3),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3))
        
        self.layer3 = nn.Sequential(
        nn.Linear(emb3, emb4),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3))
        
        self.layer_out = nn.Sequential(
        nn.Linear(emb4, out_emb),
        nn.Sigmoid())
        
    def forward(self, x):
        #print("Size before: ",x.size())
        x = self.layer1(x)
        #print("Size before 1: ",x.size())
        x = self.layer2(x)
        #print("Size before 2: ",x.size())
        x = self.layer3(x)
        #print("Size before 3: ",x.size())
        x = self.layer_out(x)
        #print("Size before out: ",x.size())
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
       
        emb2 = 42560
        out_emb = 1
        
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels =  20, kernel_size = (5,5)),
        nn.MaxPool2d(kernel_size = (2,2),stride = 2),
        nn.Dropout(0.3))
            
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels = 20,out_channels = 20, kernel_size = (5,5)),
        nn.MaxPool2d(kernel_size = (2,2),stride = 2),
        nn.Dropout(0.3),
        nn.Conv2d(in_channels = 20,out_channels = 10, kernel_size = (5,5)),
        nn.MaxPool2d(kernel_size = (2,2),stride = 2),
        nn.Dropout(0.3),
        nn.Flatten(),
         nn.Tanh())
        
        
        
    def forward(self, x):
        
        #print("Size before: ",x.size())
        x = self.layer1(x)
        #print("Size before 1: ",x.size())
        x = self.layer2(x)
        #print("Size before 2: ",x.size())
      
        return x


class KeyPointDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        ip_emb = 42560
        emb1 = 256
        emb2 = 512
        emb3 = 1024
        out_emb = 256
        
        self.layer1 = nn.Sequential(
        nn.Linear(ip_emb, emb1),
        nn.LeakyReLU(0.2))
        
        self.layer2 = nn.Sequential(
        nn.Linear(emb1, emb2),
        nn.LeakyReLU(0.2))
        
        self.layer3 = nn.Sequential(
        nn.Linear(emb2, emb3),
        nn.LeakyReLU(0.2))
        
        self.layer_out = nn.Sequential(
        nn.Linear(emb3, out_emb))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer_out(x)
        return x