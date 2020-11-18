import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from layer import *

class ACGAN(nn.Module):
    def __init__(self, in_channels, num_classes, out_channels, nker, norm='bnorm'):
        super(ACGAN, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, 100)
        
        self.dec1 = DECBR2d(1*in_channels, 8*nker, kernel_size=4, stride=1, padding=0, norm=norm, relu=0.0, bias=False)
        self.dec2 = DECBR2d(8*nker, 4*nker, kernel_size=8, stride=4, padding=2, norm=norm, relu=0.0, bias=False)
        self.dec3 = DECBR2d(4*nker, 2*nker, kernel_size=8, stride=4, padding=2, norm=norm, relu=0.0, bias=False)
        self.dec4 = DECBR2d(2*nker, 1*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec5 = DECBR2d(1*nker, out_channels, kernel_size=4, stride=2, padding=1, norm=norm, relu=None, bias=False)
    
    
    # (H_in - 1) * stride + kernelSize - 2 * padding
    def forward(self,x,labels):
        x = torch.mul(self.label_emb(labels), x)
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        

        x=self.dec1(x)

        x=self.dec2(x)

        x=self.dec3(x)

        x=self.dec4(x)
        #print(x.size())
        x=self.dec5(x)

        
        #x = torch.tanh(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, nker, norm='bnorm'):
        super(Discriminator, self).__init__()
        self.enc1 = CBR2d(1*in_channels, 1*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc2 = CBR2d(1*nker, 2*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc3 = CBR2d(2*nker, 4*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc4 = CBR2d(4*nker, 8*nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc5 = CBR2d(8*nker, out_channels, kernel_size=4, stride=2, padding=1, norm=norm, relu=None, bias=False)
        
        self.adv_layer = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.aux_layer = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x=self.enc1(x)
        x=self.enc2(x)
        x=self.enc3(x)
        x=self.enc4(x)
        x=self.enc5(x)
        
        x = x.view(-1,x.shape[-1]**2)

        return self.adv_layer(x), self.aux_layer(x)
        
        
        
        

