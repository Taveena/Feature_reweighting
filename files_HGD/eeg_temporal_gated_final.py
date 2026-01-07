####################################################################### proposed_aux (BGM after I3 module having TA + Value) #####################################################

import numpy as np
#from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import copy
import math

# https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.kernelSize = kernel_size
        self.time_padding = int(kernel_size//2)
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=(1,kernel_size), padding=(0,self.time_padding), groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(out_channels*depth, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# https://github.com/LIKANblk/AML_EEG_challenge/blob/master/src/model_torch.py
class MI_EEGNet_stem(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self, Chans=64, Samples=80,
           dropoutRates=(0.25,0.25), kernLength1=16,kernLength2=16, kernLength3=5, poolKern1=4,poolKern2=8, F1=64,
           D=4, norm_rate=0.25, dropoutType='Dropout'):
        super(MI_EEGNet_stem,self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        #block1

        time_padding = int((kernLength1//2))
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=F1,kernel_size =(1,kernLength1),padding=(0,time_padding), stride=1,bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=F1, affine=True)

        self.depthwise1 = nn.Conv2d(in_channels=F1,out_channels=F1*D,kernel_size=(Chans,1),groups=F1,padding=0,bias=False)

        self.batchnorm2 = nn.BatchNorm2d(num_features=F1*D, affine=True)
        self.activation_block1 = nn.ELU()

        self.avg_pool_block1 = nn.AdaptiveAvgPool2d((1,Samples//2))
        self.dropout_block1 = nn.Dropout(p=dropoutRates[0])
        
    def forward(self,input):
        out_dims = {}
        #input = batch size x 1 x channel x signal length
        block1 = self.conv1(input)
        block1 = self.batchnorm1(block1)
        block1 = self.depthwise1(block1)
        block1 = self.batchnorm2(block1)
        block1 = self.activation_block1(block1)
        block1 = self.avg_pool_block1(block1)
        block1 = self.dropout_block1(block1)
        #print(block1.shape)
        return block1

class ChannelSSLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """
    #### This model is for computing channel attention score ############ 
    def __init__(self, num_channels, out_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSSLayer, self).__init__()
        out_channels_reduced = out_channels // reduction_ratio
        self.conv = nn.Conv2d(num_channels, out_channels, 1,bias=False)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(out_channels, out_channels_reduced, bias=False)
        self.fc2 = nn.Linear(out_channels_reduced, out_channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #self.output = nn.LSTM(256,256,1)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        conv = self.conv(input_tensor)
        batch_size, num_channels, H, W = conv.size()
        # Average along each channel
        squeeze_tensor = conv.view(batch_size, num_channels, -1).mean(dim=2)
        #lstm_tensor = input_tensor.squeeze(-2).permute(2,0,1)
        #output, (h,c)= self.output(lstm_tensor)
        #check shape of output
        #print('output.shape',output.shape, 'hc.shape', hc[1].shape)
        #output = output.permute(1,2,0)
        #squeeze_tensor = squeeze_tensor - output[-1,:,:]
        #h = h.squeeze(0)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        excited_tensor = fc_out_2.view(a, b, 1, 1)  
        #output_tensor = torch.mul(input_tensor, excited_tensor)
        return excited_tensor

class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """
    #### This model is for computing temporal attention score ############ 
    def __init__(self, num_channels, Samples, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        #out_channels_reduced = out_channels // reduction_ratio
        self.conv1 = nn.Conv2d(num_channels, num_channels//4, 1,bias=False)
        self.avg_pool_block1 = nn.AdaptiveAvgPool2d((1,Samples//5)) 
        self.upsample1 = nn.Upsample(scale_factor = (1,2.5), mode = 'nearest')
        self.avg_pool_block2 = nn.AdaptiveAvgPool2d((1,Samples//10))
        self.upsample2 = nn.Upsample(scale_factor = (1,5), mode = 'nearest')
        self.avg_pool_block3 = nn.AdaptiveAvgPool2d((1,Samples//20)) 
        self.upsample3 = nn.Upsample(scale_factor = (1,10), mode = 'nearest')
        #self.conv2 = nn.Conv2d(out_channels, out_channels_reduced, 1,bias=False)
        self.conv3 = nn.Conv2d(num_channels, 1, 1,bias=False)
        self.sigmoid = nn.Sigmoid()#Softmax(dim=-1)#

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()
        conv_out = F.elu(self.conv1(input_tensor))		# added activation 
        #print('conv_out', conv_out.shape)
        block1 = self.avg_pool_block1(conv_out)
        #print('block1', block1.shape)
        block1 = self.upsample1(block1)
        block2 = self.avg_pool_block2(conv_out)
        #print('block2', block2.shape)
        block2 = self.upsample2(block2)
        block3 = self.avg_pool_block3(conv_out)
        #print('block2', block2.shape)
        block3 = self.upsample3(block3)
        #block1 = self.conv2(block1)
        #block2 = self.conv2(block2)
        #block3 = self.conv2(block3)
        #conv_out = self.conv2(conv_out)
        #print('upsample1', block1.shape, 'upsample2', block2.shape, 'conv_out', conv_out.shape)
        out = torch.cat([block1, block2, block3, conv_out], axis = 1)
        out = self.conv3(out)
        #out = self.score(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        #print('tensor', squeeze_tensor.shape,squeeze_tensor[0,0,0,:])
        return squeeze_tensor


class SpatialChannelSELayer(nn.Module):
    def __init__(self, num_channels, Samples):
        """
        :param num_channels: No of input channels
        """
        super(SpatialChannelSELayer, self).__init__()
        #self.channelSE=ChannelSSLayer(num_channels, out_channels) 
        self.spatialSE=SpatialSELayer(num_channels, Samples)
        #self.value = nn.Conv2d(num_channels, out_channels, 1,bias=False)

    def forward(self, input_tensor):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        #print('input', input.shape)

        #value = F.elu(self.value(input_tensor))		# append activation function 
        batch_size, channel, a, b = input_tensor.size()
        #channel_wise = self.channelSE(input_tensor).repeat(1,1,a,1)
        temporal_wise = self.spatialSE(input_tensor).repeat(1,channel, 1,1)
        #print(temporalwise.shape)
        #aux_tensor = torch.mean((input_tensor*temporalwise).squeeze(dim=-2),dim=-1)
        #temporalwise = temporalwise.repeat(1,channel, 1,1)
        #attention_map = channel_wise* input_tensor#temporalwise
        output_tensor= input_tensor*temporal_wise
        return output_tensor#, aux_tensor


class MI_EEGNet_branch(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,Chans=64, Samples=128,
           dropoutRates=0.25, kernLength1=1,kernLength2=7, poolKern1=4,poolKern2=8, F=64,
           norm_rate=0.25, dropoutType='Dropout'):
        super(MI_EEGNet_branch,self).__init__()
        self.Chans = Chans
        self.Samples = Samples

        time_padding = int((kernLength1//2))
        self.conv1 = nn.Conv2d(in_channels=Chans,out_channels=F,kernel_size =(1,kernLength1),padding=(0,time_padding), stride=1,bias=False)#out_channels=F
        
        self.separable_block1 = SeparableConv2d(in_channels=F, out_channels=F,depth=1, kernel_size=kernLength2)#in_channels=F, out_channels=F,

        self.batchnorm1 = nn.BatchNorm2d(num_features=F, affine=True)#num_features=F
        self.activation_block1 = nn.ELU()
        
        self.dropout_block1 = nn.Dropout(p=dropoutRates)

        self.separable_block2 = SeparableConv2d(in_channels=F, out_channels=F,depth=1, kernel_size=kernLength2)#in_channels=F,  out_channels=F

        self.avg_pool_block1 = nn.AdaptiveAvgPool2d((1,Samples//2))
        #self.channeltemporalSELayer1 = SpatialChannelSELayer(F,40)#F
        #self.channeltemporalSELayer2 = SpatialChannelSELayer(F,40)#F

    def forward(self,input):
        block1 = self.conv1(input)
        block1 = self.separable_block1(block1)
                #block1, aux1 = self.channeltemporalSELayer1(block1)
        #block1 = self.ChannelSSLayer1(block1)

        block1 = self.batchnorm1(block1)
        block1 = self.activation_block1(block1)
        block1 = self.dropout_block1(block1)
        block1 = self.separable_block2(block1)
                #block1, aux2 = self.channeltemporalSELayer2(block1)
        #block1 = self.ChannelSSLayer2(block1)
        block1 = self.avg_pool_block1(block1)
        return block1 #, aux1+aux2
  
class Inception(nn.Module):
    def __init__(self,Chans=64, Samples=256,
           dropoutRates=(0.25), kernLength1=[1,1],kernLength2=[7,9], poolKern1=[4,4],poolKern2=[8,8], F=64, norm_rate=[0.25,0.25], dropoutType='Dropout'): 
        super(Inception,self).__init__()
        self.output_sizes = {}
        self.branch1 = MI_EEGNet_branch(Chans=Chans, Samples=Samples, dropoutRates=dropoutRates, kernLength1=kernLength1[0],kernLength2=kernLength2[0], poolKern1=poolKern1[0],poolKern2=poolKern2[0],
                                        F=F, norm_rate=norm_rate[0], dropoutType='Dropout')

        self.branch2 = MI_EEGNet_branch(Chans=Chans, Samples=Samples, dropoutRates=dropoutRates, kernLength1=kernLength1[1],kernLength2=kernLength2[1], poolKern1=poolKern1[1],poolKern2=poolKern2[1],
                                        F=F, norm_rate=norm_rate[1], dropoutType='Dropout')

        self.avgpool1 = nn.AdaptiveAvgPool2d((1,Samples//2))
        self.conv3_1 = nn.Conv2d(in_channels=Chans,out_channels=F,kernel_size =(1,1), stride=1,bias=False)#out_channels=64

        self.conv4_1 = nn.Conv2d(in_channels=Chans,out_channels=F,kernel_size =(1,1),padding=(0,0), stride=(1,2), bias=False)#out_channels=64
        self.conv = nn.Conv2d(Chans, Chans//2, 1,bias=False) 
        self.channeltemporalSELayer = SpatialChannelSELayer(Chans//2,Samples)

    def forward(self, input):

        block1 = self.branch1(input) #, aux1 
        block2 = self.branch2(input) #, aux2 
        block3 = self.conv3_1(self.avgpool1(input))
        #print('input1', input.shape)
        block4 = self.conv4_1(input)
        #print('input2', input.shape, 'block4', block4.shape)
        #print(block1.shape, block2.shape, block3.shape, block4.shape)
        add = torch.cat([block1, block2, block3, block4], axis=1)
        block5 = F.elu(self.conv(add))
        block5  = self.channeltemporalSELayer(block5)
        #print('block5 shape', block5.shape)

        return block5
'''
class Inception(nn.Module):
    def __init__(self, Chans=64, Samples=128): 
        super(Inception, self).__init__()

        self.inception1 = Inception_module(Chans=Chans, Samples=Samples)
        self.inception2 = Inception_module(Chans=Chans, Samples=Samples)
        self.inception3 = Inception_module(Chans=Chans, Samples=Samples)
        self.inception4 = Inception_module(Chans=Chans, Samples=Samples)
        self.conv = nn.Conv2d(Chans, Chans//2, 1,bias=False) 
        self.channeltemporalSELayer = SpatialChannelSELayer(Chans//2,Chans//4,Samples)

    def forward(self, input):

        incep1 = self.inception1(input) 
        incep2 = self.inception2(input)
        incep3 = self.inception3(input)
        incep4 = self.inception4(input)
        add = incep1+ incep2+ incep3+ incep4
        block5 = F.elu(self.conv(add))
        block5  = self.channeltemporalSELayer(block5)

        return block5 #add , aux   #, [aux1, aux2]
'''
    
class MI_EEGNet_predict(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,nb_classes, Chans=256, Samples=20,
           dropoutRates=0.25, kernLength1=5,F=256, F1=128, D=4, norm_rate=0.25, dropoutType='Dropout'):
        super(MI_EEGNet_predict,self).__init__()
        self.Chans = Chans
        self.Samples = Samples

        self.batchnorm1 = nn.BatchNorm2d(num_features=F1, affine=True)
        self.activation_block1 = nn.ELU()

        self.separable_block = SeparableConv2d(in_channels=F1, out_channels=F1 ,depth=1, kernel_size=kernLength1)

        self.batchnorm2 = nn.BatchNorm2d(num_features=F1, affine=True)
        self.activation_block2 = nn.ELU()

        self.avg_pool_block = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = nn.Dropout(dropoutRates)

        self.dense = nn.Linear(128,nb_classes)
        #self.aux_out = nn.Linear(256,nb_classes)#64,nb_classes
        #self.aux2_out = nn.Linear(64,nb_classes)#64,nb_classes

    def forward(self, input):     #,aux

        block1 = self.batchnorm1(input)
        block1 = self.activation_block1(block1)
        block1 = self.separable_block(block1)

        block2 = self.batchnorm2(block1)
        block2 = self.activation_block2(block2)
        block2 = self.dropout(block2)
        features = self.avg_pool_block(block2)

        batch_size = features.shape[0]
        flatten_feats = features.reshape(batch_size,-1)
        out = self.dense(flatten_feats)
        #aux = self.aux_out(aux)
        #aux2 = self.aux2_out(aux[1])
        return out, flatten_feats#, aux

class EEGNet_proposed_temporal_gated_final(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,nb_classes, Chans=64, Samples=1000, moabb=False):     ###### moabb=False added for braindecode moabb
        super(EEGNet_proposed_temporal_gated_final,self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        self.moabb = moabb     ###### added for braindecode moabb
        self.stem = MI_EEGNet_stem(Chans=Chans, Samples=Samples)
        self.inception = Inception(Chans=256, Samples=Samples//2)#Chans=256, Samples=Samples//2
        self.prediction = MI_EEGNet_predict(nb_classes=nb_classes, Chans=256, Samples=Samples//4)#Chans=256

    def forward(self, input, embedding=False):
        #print('input shape', input.shape)
        input = input.unsqueeze(dim=1)     ###### added for braindecode moabb
        out = self.stem(input) 
        #print('out.shape stem', out.shape)
        #print('out shape', out.shape)
        out = self.inception(out) #, aux
        #print('out.shape inception', out.shape, 'aux shape', len(aux))
        out, features = self.prediction(out) #, aux
        #print('out.shape prediction', out.shape, 'aux1', len(aux1), 'aux2', len(aux2))
        if self.moabb==True:     ###### added for braindecode moabb
             out = out.unsqueeze(dim=-1)     ###### added for braindecode moabb
             out = F.softmax(out, dim=-2)     ###### added for braindecode moabb
        if embedding:
            return out, features
        else:
            return out

    def weights_init(self):
        def weights_init(m):
             if isinstance(m, nn.Conv2d):
                 torch.nn.init.xavier_uniform(m.weight.data)
             if isinstance(m,nn.Linear):
                 torch.nn.init.xavier_uniform(m.weight.data)
                 m.bias.data.fill_(0.01) 

if __name__=='__main__':
     net=EEGNet_proposed_temporal_gated_final(1, Chans=22, Samples=1000)
     input=torch.randn(5,1,22,1000)
     output = net(input)
     from torchinfo import summary
     summary(net.cuda(), (5, 1,22, 1000))
     print(output[0].shape)
