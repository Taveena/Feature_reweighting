############################################################ proposed_aux #############################################

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

'''
# https://gist.github.com/bdsaglam/b16de6ae6662e7a783e06e58e2c5185a
class DepthwiseConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, depth_multiplier=1, kernel_size=3, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        out_channels = in_channels * depth_multiplier
        super().__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=in_channels,bias=bias,
            padding_mode=padding_mode)
'''

# https://github.com/LIKANblk/AML_EEG_challenge/blob/master/src/model_torch.py
class MI_EEGNet_stem(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self, Chans=64, Samples=80,
           dropoutRates=(0.25,0.25), kernLength1=16, kernLength2=16, kernLength3=5, poolKern1=4,poolKern2=8, F1=64,
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

class MI_EEGNet_branch(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,Chans=64, Samples=128,
           dropoutRates=0.25, kernLength1=1,kernLength2=7, poolKern1=4,poolKern2=8, F=64, F1=32, #changed value of number of filter to separable conv to reduce the parameters to avoid overfitting
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

        self.separable_block2 = SeparableConv2d(in_channels=F, out_channels=F,depth=1, kernel_size=kernLength2)#in_channels=F,  out_channels=F1

        self.avg_pool_block1 = nn.AdaptiveAvgPool2d((1,Samples//2))
        #self.channeltemporalSELayer1 = SpatialChannelSELayer(Chans,Chans//2,Samples) #(Chans//4)*3,
        #self.channeltemporalSELayer2 = SpatialChannelSELayer(Chans,Chans//2,Samples) #,(Chans//4)*3

    def forward(self,input):
        block1 = self.conv1(input)
        block1 = self.separable_block1(block1)
        #block1, aux1 = self.channeltemporalSELayer1(block1)
        #block1 = self.channeltemporalSELayer1(block1)
        #block1 = self.ChannelSSLayer1(block1)

        block1 = self.batchnorm1(block1)
        block1 = self.activation_block1(block1)
        block1 = self.dropout_block1(block1)
        block1 = self.separable_block2(block1)
        #block1, aux2 = self.channeltemporalSELayer2(block1)
        #block1 = self.channeltemporalSELayer2(block1)
        #block1 = self.ChannelSSLayer2(block1)
        block1 = self.avg_pool_block1(block1)
        return block1#, aux1+aux2


class ChannelSSLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSSLayer, self).__init__()
        num_channels_reduced1 = num_channels // 4
        num_channels_reduced2 = num_channels // 8
        #self.reduction_ratio = reduction_ratio
        #self.fc1 = nn.Linear(num_channels, num_channels//2, bias=True)
        self.fc1_1 = nn.Linear((num_channels_reduced1), num_channels_reduced1//2, bias=True)
        self.fc1_2 = nn.Linear((num_channels_reduced1), num_channels_reduced1//2, bias=True)
        self.fc1_3 = nn.Linear((num_channels_reduced1), num_channels_reduced1//2, bias=True)
        self.fc1_4 = nn.Linear((num_channels_reduced1), num_channels_reduced1//2, bias=True)
        #self.fc1 = nn.Linear(num_channels_reduced1, num_channels_reduced1//2, bias=True)
        self.fc2 = nn.Linear(4*num_channels_reduced1//2, num_channels_reduced2, bias=True)
        self.fc3 = nn.Linear(num_channels_reduced2, num_channels, bias=True)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=-1)#Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        #print('SS Layer', input_tensor.shape)
        batch_size, num_channels, H, W = input_tensor.size()
        #print('input size', input_tensor.shape)
        #squeeze_tensor = self.fc1(input_tensor)
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        squeeze_tensor1 = squeeze_tensor[:,0:(num_channels//4)]
        squeeze_tensor2 = squeeze_tensor[:,(num_channels//4):(num_channels//2)]
        squeeze_tensor3 = squeeze_tensor[:,(num_channels//2):(3*num_channels//4)]
        squeeze_tensor4 = squeeze_tensor[:,(3*num_channels//4):]
        #print('squeeze tensor', squeeze_tensor1.shape, squeeze_tensor2.shape, squeeze_tensor3.shape)
        # channel excitation
        tensor_a = self.relu(self.fc1_1(squeeze_tensor1))
        tensor_b = self.relu(self.fc1_2(squeeze_tensor2))
        tensor_c = self.relu(self.fc1_3(squeeze_tensor3))
        tensor_d = self.relu(self.fc1_4(squeeze_tensor4))
        #print('shape of a, b, c, d', tensor_a.shape, tensor_b.shape, tensor_c.shape, tensor_d.shape)
        fc_out_1 = torch.cat([tensor_a, tensor_b, tensor_c, tensor_d], axis=1)
        fc_out_2 = self.relu(self.fc2(fc_out_1))
        fc_out_3 = self.fc3(fc_out_2)
        #print('shape of fc_out_1', fc_out_1.shape, 'fc_out_2', fc_out_2.shape, 'fc_out_3', fc_out_3.shape)

        a, b = squeeze_tensor.size()
        #print('input tensor shape', input_tensor.shape, 'squeeze tensor shape', squeeze_tensor.shape)
        excited_tensor = fc_out_3.view(a, b, 1, 1)
        #print('input tensor shape', input_tensor.shape, 'excited tensor shape', excited_tensor.shape)  
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

        self.conv1 = nn.Conv2d(num_channels, num_channels//4, 1,bias=False)
        self.avg_pool_block1 = nn.AdaptiveAvgPool2d((1,Samples//5)) 
        self.upsample1 = nn.Upsample(scale_factor = (1,2.5), mode = 'nearest')
        self.avg_pool_block2 = nn.AdaptiveAvgPool2d((1,Samples//10))
        self.upsample2 = nn.Upsample(scale_factor = (1,5), mode = 'nearest')
        self.avg_pool_block3 = nn.AdaptiveAvgPool2d((1,Samples//20)) 
        self.upsample3 = nn.Upsample(scale_factor = (1,10), mode = 'nearest')
        self.conv3 = nn.Conv2d(num_channels, 1, 1,bias=False)
        #self.softmax = nn.Softmax(dim=-1)#Sigmoid()#

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()
        #print('temporal length', input_tensor.shape)
        conv_out = F.elu(self.conv1(input_tensor))		# added activation 
        #print('conv_out', conv_out.shape)
        block1 = self.avg_pool_block1(conv_out)
        #print('block1', block1.shape)
        block1 = self.upsample1(block1)
        #print('block1', block1.shape)
        block2 = self.avg_pool_block2(conv_out)
        #print('block2', block2.shape)
        block2 = self.upsample2(block2)
        #print('block2', block2.shape)
        block3 = self.avg_pool_block3(conv_out)
        #print('block2', block2.shape)
        block3 = self.upsample3(block3)
        #block1 = self.conv2(block1)
        #block2 = self.conv2(block2)
        #block3 = self.conv2(block3)
        #conv_out = self.conv2(conv_out)
        #print('upsample1', block1.shape, 'upsample2', block2.shape, 'conv_out', conv_out.shape)
        out = torch.cat([block1, block2, block3, conv_out], axis = 1)
        squeeze_tensor = self.conv3(out)
        #out = self.score(input_tensor)

        #squeeze_tensor = self.softmax(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        #print('tensor', squeeze_tensor.shape,squeeze_tensor[0,0,0,:])
        return squeeze_tensor

class sknet_module(nn.Module):
    def __init__(self, num_channels, Samples):
        """
        :param num_channels: No of input channels
        """
        super(sknet_module, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, (1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=num_channels, affine=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, (1,3), bias=False, padding="same")
        self.bn2 = nn.BatchNorm2d(num_features=num_channels, affine=True)
        self.channelSE=ChannelSSLayer(num_channels) 
        self.spatialSE=SpatialSELayer(num_channels, Samples)
        #self.value = nn.Conv2d(num_channels, num_channels, 1,bias=False)
        #self.combine = nn.Conv2d(num_channels*2, num_channels, 1,bias=False)
        #self.bn3 = nn.BatchNorm2d(num_features=num_channels, affine=True)
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU()

    def forward(self, input_tensor):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        #print('input', input.shape)
        batch_size, channel, a, b = input_tensor.size()
        

        block1 = self.elu(self.bn1(self.conv1(input_tensor)))
        block2 = self.elu(self.bn2(self.conv2(input_tensor)))
        #print('block1 and block2 shape', block1.shape, block2.shape)
        add = block1 +block2 
        #value = F.elu(self.value(input_tensor))   
        batch_size, channel, a, b = input_tensor.size()  
        #print('value shape', value.shape, 'batch_size, channel, a, b', batch_size, channel, a, b)   
        channelwise = self.channelSE(add).repeat(1,1,1,b)
        temporalwise = self.spatialSE(add).repeat(1,channel, 1,1)
        #print('temporalwise shape', temporalwise.shape, 'channelwise shape', channelwise.shape)

        #concat = torch.cat([channelwise, temporalwise], axis=1)
        concat = torch.cat([channelwise.permute(0,2,1,3), temporalwise.permute(0,2,1,3)],axis=1)
        concat = self.softmax(concat)
        #print('dimension of block1 and block2', block1.shape, block2.shape, 'concat shape', concat.shape)

        block1 = block1 * concat[:,0,:,:].unsqueeze(2)
        block2 = block2 * concat[:,1,:,:].unsqueeze(2)
        #output_tensor = self.combine(torch.cat([block1, block2], dim=1))
        output_tensor = block1*block2

        return output_tensor#, aux_tensor


class Inception(nn.Module):
    def __init__(self,Chans=64, Samples=128,
           dropoutRates=(0.25), kernLength1=[1,1],kernLength2=[7,9], poolKern1=[4,4],poolKern2=[8,8], F=64, norm_rate=[0.25,0.25], dropoutType='Dropout'): 
        super(Inception,self).__init__()
        self.output_sizes = {}
        self.branch1 = MI_EEGNet_branch(Chans=Chans, Samples=Samples, dropoutRates=dropoutRates, kernLength1=kernLength1[0],kernLength2=kernLength2[0], poolKern1=poolKern1[0],poolKern2=poolKern2[0],
                                        F=F, norm_rate=norm_rate[0], dropoutType='Dropout')

        self.branch2 = MI_EEGNet_branch(Chans=Chans, Samples=Samples, dropoutRates=dropoutRates, kernLength1=kernLength1[1],kernLength2=kernLength2[1], poolKern1=poolKern1[1],poolKern2=poolKern2[1],
                                        F=F, norm_rate=norm_rate[1], dropoutType='Dropout')

        self.avgpool1 = nn.AdaptiveAvgPool2d((1,Samples//2))
        self.conv3_1 = nn.Conv2d(in_channels=Chans,out_channels=64,kernel_size =(1,1), stride=1,bias=False)#out_channels=64

        self.conv4_1 = nn.Conv2d(in_channels=Chans,out_channels=64,kernel_size =(1,1),padding=(0,0), stride=(1,2),bias=False)#out_channels=64
        self.conv5 = nn.Conv2d(Chans, Chans//2, 1,bias=False)
        #self.channeltemporalSELayer = SpatialChannelSELayer(Chans//2,Samples)
        self.sknet = sknet_module(Chans//2,Samples)

    def forward(self, input):

        block1 = self.branch1(input) #, aux1 
        block2 = self.branch2(input) #, aux2
        block3 = self.conv3_1(self.avgpool1(input))
        block4 = self.conv4_1(input)
        #print(block1.shape, block2.shape, block3.shape, block4.shape)
        add = torch.cat([block1, block2, block3, block4], axis=1)
        block5 = F.elu(self.conv5(add))
        block5  = self.sknet(block5)
        #print('block5', block5.shape)

        return block5   #, [aux1, aux2]

    
class MI_EEGNet_predict(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,nb_classes, Chans=256, Samples=20,
           dropoutRates=0.25, kernLength1=5,F=256,F1=128, D=4, norm_rate=0.25, dropoutType='Dropout'):
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
        #self.aux1_out = nn.Linear(64,nb_classes)#64,nb_classes
        #self.aux2_out = nn.Linear(64,nb_classes)#64,nb_classes

    def forward(self,input): #,aux):

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
        #aux1 = self.aux1_out(aux[0])
        #aux2 = self.aux2_out(aux[1])
        return out,flatten_feats#, aux1, aux2

class EEGNet_fgnet(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,nb_classes, Chans=64, Samples=1000):
        super(EEGNet_fgnet,self).__init__()
        self.Chans = Chans
        self.Samples = Samples
    
        self.stem = MI_EEGNet_stem(Chans=Chans, Samples=Samples)
        self.inception = Inception(Chans=256, Samples=Samples//2)#Chans=256, Samples=Samples//2
        self.prediction = MI_EEGNet_predict(nb_classes=nb_classes, Chans=256, Samples=Samples//4)#Chans=256

    def forward(self, input, embedding=False):
        out = self.stem(input) 
        #print('out.shape stem', out.shape)
        out = self.inception(out)  #, aux
        #print('out.shape inception', out.shape, 'aux shape', len(aux))
        out, features = self.prediction(out)  #  , aux            #, aux1, aux2 
        #print('out.shape prediction', out.shape, 'aux1', len(aux1), 'aux2', len(aux2))
        if embedding:
            return out, features#, aux1, aux2
        else:
            return out#, aux1, aux2 

    def weights_init(self):
        def weights_init(m):
             if isinstance(m, nn.Conv2d):
                 torch.nn.init.xavier_uniform(m.weight.data)
             if isinstance(m,nn.Linear):
                 torch.nn.init.xavier_uniform(m.weight.data)
                 m.bias.data.fill_(0.01) 

if __name__=='__main__':
     net=EEGNet_fgnet(1, Chans=22, Samples=1000)
     input=torch.randn(5,1,22,1000)
     output = net(input)
     from torchinfo import summary
     summary(net.cuda(), (5, 1,22, 1000))
     print(output[0].shape)



