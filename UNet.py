import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import random
import torchvision

class Conv2DReBn(nn.Module):
    
    def __init__(self, in_channels, out_channels, padding, hyperparameters):
        
        super().__init__()

        nonlinearity = hyperparameters['nonlinear_activation']
        dropout_rate = hyperparameters['dropout_rate']

        self.conv = nn.Conv2d(in_channels, out_channels,3, padding = padding)

        if nonlinearity == 'relu':
            self.nonlinear = nn.ReLU()
        elif nonlinarity == 'elu':
            self.nonlinear = nn.ELU()
        else:
            raise Exception('hyperparameters must be relu or elu')

        self.bn = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
    def forward(self, x):
        
        return self.dropout(self.bn(self.nonlinear(self.conv(x))))

    
class EncodeStack(nn.Module):
        
    def __init__(self, in_channels, out_channels, padding, hyperparameters):
        
        super().__init__()
        self.convbnre1 = Conv2DReBn(in_channels, out_channels, padding, hyperparameters)
        self.convbnre2 = Conv2DReBn(out_channels, out_channels, padding, hyperparameters)
        self.pool = nn.MaxPool2d(2, stride = 2)
    
    def forward(self, x):
        
        encoded_features = self.convbnre2(self.convbnre1(x)) 
        
        return self.pool(encoded_features), encoded_features

    
class DecodeStack(nn.Module):
    
    def __init__(self, in_channels, out_channels, padding, hyperparameters):
        
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, (2,2), stride = 2)
        self.convbnre1 = Conv2DReBn(in_channels, out_channels, padding, hyperparameters)
        self.convbnre2 = Conv2DReBn(out_channels, out_channels, padding, hyperparameters)
    
    
    def trim_tensor(self, to_trim, size):
        original_dim = int(to_trim.size()[2])
        border = int((original_dim - size)/2)
        return to_trim[:,:,border:original_dim-border, border:original_dim-border]
        
    
    def forward(self, x, encoded_features):
        
        x = self.trans_conv(x)
        x = self.convbnre1(torch.cat((self.trim_tensor(encoded_features, x.size()[2]), x),1))
        
        #x = self.convbnre1(torch.cat((encoded_features, x),1))
        x = self.convbnre2(x)
        
        return x
        
        
class UNet(nn.Module):
    
    def __init__(self, hyperparameters):
        
        super().__init__()

        depth = hyperparameters['model_depth']

        output_size_same = True
        if output_size_same:
            padding = [1,1]
        else:
            padding = 0

        self.encode1 = EncodeStack(3, 64, padding, hyperparameters)
        self.encode2 = EncodeStack(64, 128, padding, hyperparameters)
        self.encode3 = EncodeStack(128, 256, padding, hyperparameters)
        self.encode4 = EncodeStack(256, 512, padding, hyperparameters)
        
        self.center = nn.Sequential(
            Conv2DReBn(512,1024, padding, hyperparameters),
            Conv2DReBn(1024,1024, padding, hyperparameters)
        )
        
        self.decode1 = DecodeStack(1024, 512, padding, hyperparameters)
        self.decode2 = DecodeStack(512, 256, padding, hyperparameters)
        self.decode3 = DecodeStack(256, 128, padding, hyperparameters)
        self.decode4 = DecodeStack(128, 64, padding, hyperparameters)
        
        self.conv = nn.Conv2d(64,1,1)
            
    
    def forward(self, x):
        
        output = x
        output, encoded_features1 = self.encode1(output)
        output, encoded_features2 = self.encode2(output)
        output, encoded_features3 = self.encode3(output)
        output, encoded_features4 = self.encode4(output)
        
        output = self.center(output)
        
        output = self.decode1(output, encoded_features4)
        output = self.decode2(output, encoded_features3)
        output = self.decode3(output, encoded_features2)
        output = self.decode4(output, encoded_features1)
        
        output = self.conv(output)
        
        return output
