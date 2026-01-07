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



############ EEGNet V1 ##########################################################
class EEGNet(nn.Module):
    def __init__(self, out_channels):
        super(EEGNet, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (64, 1), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, affine=True)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((15, 16, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, affine=True)
        self.pooling2 = nn.MaxPool2d((2, 4), stride=(2,4))
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, affine=True)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        self.fc1 = nn.Linear(4*4*5, out_channels)

        self.dropout_block1 = nn.Dropout(p=0.25)
        self.dropout_block2 = nn.Dropout(p=0.25)
        self.dropout_block3 = nn.Dropout(p=0.25)        

    def forward(self, x):
        # Layer 1
        #print(x.shape)
        x = F.elu(self.batchnorm1(self.conv1(x)))
        x = x.permute(0, 2, 1, 3)
        x = self.dropout_block1(x)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.batchnorm2(self.conv2(x)))
        x = self.pooling2(x)
        x = self.dropout_block2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.batchnorm3(self.conv3(x)))
        x = self.pooling3(x)
        x = self.dropout_block3(x)

        # FC Layer
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

    def weights_init(self):
        def weights_init(m):
             if isinstance(m, nn.Conv2d):
                 torch.nn.init.xavier_uniform(m.weight.data)
             if isinstance(m,nn.Linear):
                 torch.nn.init.xavier_uniform(m.weight.data)
                 m.bias.data.fill_(0.01) 


############ EEGNet V2 ##########################################################

def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0,dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    if type(dilation) is not tuple:
        dilation = (dilation,dilation)

    h = math.floor((h_w[0] + 2*pad[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + 2*pad[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    return h, w

def get_model_params(model):
    params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_dict[name] = param.data
    return params_dict

class deepwise_separable_conv(nn.Module):
    def __init__(self,nin,nout,kernelSize):
        super(deepwise_separable_conv,self).__init__()
        self.kernelSize = kernelSize
        self.time_padding = int(kernelSize//2)
        self.depthwise = nn.Conv2d(in_channels=nin,out_channels=nin,kernel_size=(1,kernelSize),
                                   padding=(0,self.time_padding),groups=nin,bias=False)
        self.pointwise = nn.Conv2d(in_channels=nin,out_channels=nout, kernel_size=1,groups=1,bias=False)
    def forward(self, input):
        dw = self.depthwise(input)
        pw = self.pointwise(dw)
        return pw
    def get_output_size(self,h_w):
        return convtransp_output_shape(h_w, kernel_size=(1,self.kernelSize), stride=1, pad=(0,self.time_padding), dilation=1)


# https://github.com/LIKANblk/AML_EEG_challenge/blob/master/src/model_torch.py
class EEGNet_v2(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,nb_classes, Chans=64, Samples=128,
           dropoutRates=(0.25,0.25), kernLength1=64,kernLength2=16, poolKern1=4,poolKern2=8, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet_v2,self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        self.output_sizes = {}
        #block1
        time_padding = int((kernLength1//2))
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=F1,kernel_size =(1,kernLength1),padding=(0,time_padding), stride=1,bias=False)
        self.output_sizes['conv1']=convtransp_output_shape((Chans,Samples), kernel_size=(1,kernLength1), stride=1,
                                                           pad=(0,time_padding))
        self.batchnorm1 = nn.BatchNorm2d(num_features=F1, affine=True)
        self.depthwise1 = nn.Conv2d(in_channels=F1,out_channels=F1*D,kernel_size=(Chans,1),groups=F1,padding=0,bias=False)
        self.output_sizes['depthwise1'] = convtransp_output_shape(self.output_sizes['conv1'], kernel_size=(Chans,1),
                                                                  stride=1, pad=0)
        self.batchnorm2 = nn.BatchNorm2d(num_features=F1*D, affine=True)
        self.activation_block1 = nn.ELU()
        # self.avg_pool_block1 = nn.AvgPool2d((1,poolKern1))
        # self.output_sizes['avg_pool_block1'] = convtransp_output_shape(self.output_sizes['depthwise1'], kernel_size=(1, poolKern1),
        #                                                           stride=(1,poolKern1), pad=0)
        self.avg_pool_block1 = nn.AdaptiveAvgPool2d((1,int(self.output_sizes['depthwise1'][1]/4)))
        self.output_sizes['avg_pool_block1'] = (1,int(self.output_sizes['depthwise1'][1]/4))
        self.dropout_block1 = nn.Dropout(p=dropoutRates[0])

        #block2
        self.separable_block2 = deepwise_separable_conv(nin=F1*D,nout=F2,kernelSize=kernLength2)
        self.output_sizes['separable_block2'] = self.separable_block2.get_output_size(self.output_sizes['avg_pool_block1'])
        self.activation_block2 = nn.ELU()
        # self.avg_pool_block2 = nn.AvgPool2d((1,poolKern2))
        # self.output_sizes['avg_pool_block2'] = convtransp_output_shape(self.output_sizes['separable_block2'],
        #                                                                kernel_size=(1, poolKern2),
        #                                                                stride=(1, poolKern2), pad=0)
        self.avg_pool_block2 = nn.AdaptiveAvgPool2d((1,int(self.output_sizes['separable_block2'][1]/8)))
        self.output_sizes['avg_pool_block2'] = (1,int(self.output_sizes['separable_block2'][1]/8))

        self.dropout_block2 = nn.Dropout(dropoutRates[1])

        n_size = self.get_features_dim(Chans,Samples)
        self.dense = nn.Linear(n_size,nb_classes)


    def get_features_dim(self,Chans,Samples):
        bs = 1
        x = Variable(torch.rand((bs,1,Chans, Samples)))
        output_feat,out_dims = self.forward_features(x)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward_features(self,input):
        out_dims = {}
        block1 = self.conv1(input)
        out_dims['conv1'] = block1.size()
        block1 = self.batchnorm1(block1)
        block1 = self.depthwise1(block1)
        out_dims['depthwise1'] = block1.size()
        block1 = self.batchnorm2(block1)
        block1 = self.activation_block1(block1)
        block1 = self.avg_pool_block1(block1)
        out_dims['avg_pool_block1'] = block1.size()
        block1 = self.dropout_block1(block1)

        block2 = self.separable_block2(block1)
        out_dims['separable_block2'] = block1.size()
        block2 = self.activation_block2(block2)
        block2 = self.avg_pool_block2(block2)
        out_dims['avg_pool_block2'] = block1.size()
        block2 = self.dropout_block2(block2)
        return block2, out_dims

    def forward(self, input, embedding=False):
        features,_ = self.forward_features(input)
        batch_size = features.shape[0]
        flatten_feats = features.reshape(batch_size,-1)
        out = self.dense(flatten_feats)
        if embedding==True:
            return out, flatten_feats
        else: 
            return out

    def weights_init(self):
        def weights_init(m):
             if isinstance(m, nn.Conv2d):
                 torch.nn.init.xavier_uniform(m.weight.data)
             if isinstance(m,nn.Linear):
                 torch.nn.init.xavier_uniform(m.weight.data)
                 m.bias.data.fill_(0.01) 

############################### EEGNet Fusion###############################

class EEGNet_fusion_base(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,nb_classes, Chans=64, Samples=128,
           dropoutRates=(0.25,0.25), kernLength1=64,kernLength2=16, poolKern1=4,poolKern2=8, F1=4,
           D=2, F2=8, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet_fusion_base,self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        self.output_sizes = {}
        #block1
        time_padding = int((kernLength1//2))
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=F1,kernel_size =(1,kernLength1),padding=(0,time_padding), stride=1,bias=False)
        self.output_sizes['conv1']=convtransp_output_shape((Chans,Samples), kernel_size=(1,kernLength1), stride=1,
                                                           pad=(0,time_padding))
        self.batchnorm1 = nn.BatchNorm2d(num_features=F1, affine=True)
        self.depthwise1 = nn.Conv2d(in_channels=F1,out_channels=F1*D,kernel_size=(Chans,1),groups=F1,padding=0,bias=False)
        self.output_sizes['depthwise1'] = convtransp_output_shape(self.output_sizes['conv1'], kernel_size=(Chans,1),
                                                                  stride=1, pad=0)
        self.batchnorm2 = nn.BatchNorm2d(num_features=F1*D, affine=True)
        self.activation_block1 = nn.ELU()
        # self.avg_pool_block1 = nn.AvgPool2d((1,poolKern1))
        # self.output_sizes['avg_pool_block1'] = convtransp_output_shape(self.output_sizes['depthwise1'], kernel_size=(1, poolKern1),
        #                                                           stride=(1,poolKern1), pad=0)
        self.avg_pool_block1 = nn.AdaptiveAvgPool2d((1,int(self.output_sizes['depthwise1'][1]/4)))
        self.output_sizes['avg_pool_block1'] = (1,int(self.output_sizes['depthwise1'][1]/4))
        self.dropout_block1 = nn.Dropout(p=dropoutRates[0])

        #block2
        self.separable_block2 = deepwise_separable_conv(nin=F1*D,nout=F2,kernelSize=kernLength2)
        self.output_sizes['separable_block2'] = self.separable_block2.get_output_size(self.output_sizes['avg_pool_block1'])
        self.activation_block2 = nn.ELU()
        # self.avg_pool_block2 = nn.AvgPool2d((1,poolKern2))
        # self.output_sizes['avg_pool_block2'] = convtransp_output_shape(self.output_sizes['separable_block2'],
        #                                                                kernel_size=(1, poolKern2),
        #                                                                stride=(1, poolKern2), pad=0)
        self.avg_pool_block2 = nn.AdaptiveAvgPool2d((1,int(self.output_sizes['separable_block2'][1]/8)))
        self.output_sizes['avg_pool_block2'] = (1,int(self.output_sizes['separable_block2'][1]/8))

        self.dropout_block2 = nn.Dropout(dropoutRates[1])

        n_size = self.get_features_dim(Chans,Samples)
#        self.dense = nn.Linear(n_size,nb_classes)


    def get_features_dim(self,Chans,Samples):
        bs = 1
        x = Variable(torch.rand((bs,1,Chans, Samples)))
        output_feat,out_dims = self.forward_features(x)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward_features(self,input):
        out_dims = {}
        #print(input.shape)
        block1 = self.conv1(input)
        out_dims['conv1'] = block1.size()
        block1 = self.batchnorm1(block1)
        block1 = self.depthwise1(block1)
        out_dims['depthwise1'] = block1.size()
        block1 = self.batchnorm2(block1)
        block1 = self.activation_block1(block1)
        block1 = self.avg_pool_block1(block1)
        out_dims['avg_pool_block1'] = block1.size()
        block1 = self.dropout_block1(block1)

        block2 = self.separable_block2(block1)
        out_dims['separable_block2'] = block1.size()
        block2 = self.activation_block2(block2)
        block2 = self.avg_pool_block2(block2)
        out_dims['avg_pool_block2'] = block1.size()
        block2 = self.dropout_block2(block2)
        #print('block2 shape',block2.shape)
        return block2, out_dims

    def forward(self, input):
        #print('input',input.shape)
        features,_ = self.forward_features(input)
        #print('features',features.shape)
        batch_size = features.shape[0]
        flatten_feats = features.reshape(batch_size,-1)
        #print(flatten_feats.shape)
        #out = self.dense(flatten_feats)
        return flatten_feats


class EEGNet_fusion(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    def __init__(self,nb_classes,Chans=64, Samples=128):
        super(EEGNet_fusion,self).__init__()
        self.branch1 = EEGNet_fusion_base(nb_classes, Chans=Chans, Samples=Samples,
           dropoutRates=(0.25,0.25), kernLength1=64,kernLength2=8, poolKern1=4,poolKern2=8, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
        self.branch2 = EEGNet_fusion_base(nb_classes, Chans=Chans, Samples=Samples,
           dropoutRates=(0.25,0.25), kernLength1=96,kernLength2=16, poolKern1=4,poolKern2=8, F1=16,
           D=2, F2=32, norm_rate=0.25, dropoutType='Dropout')
        self.branch3 = EEGNet_fusion_base(nb_classes, Chans=Chans, Samples=Samples,
           dropoutRates=(0.25,0.25), kernLength1=128,kernLength2=32, poolKern1=4,poolKern2=8, F1=32,
           D=2, F2=64, norm_rate=0.25, dropoutType='Dropout')
        n_size=3472
        self.dense = nn.Linear(n_size,nb_classes)

    def forward(self, input, embedding=False):
        #print('out1', input.shape)
        out1 = self.branch1(input)
        #print('out1', out1.shape)
        out2 = self.branch2(input)
        out3 = self.branch3(input)
        #print('out1.shape',out1.shape,'out2.shape', out2.shape,'out3.shape', out3.shape)
        fusion = torch.cat([out1, out2, out3],dim=1)
        #print(fusion.shape)	
        out = self.dense(fusion)
        if embedding==True:
            return out, fusion
        else: 
            return out
        return out


    def weights_init(self):
        def weights_init(m):
             if isinstance(m, nn.Conv2d):
                 torch.nn.init.xavier_uniform(m.weight.data)
             if isinstance(m,nn.Linear):
                 torch.nn.init.xavier_uniform(m.weight.data)
                 m.bias.data.fill_(0.01) 


################### MI-EEGNet ###############################################

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


def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0,dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    if type(dilation) is not tuple:
        dilation = (dilation,dilation)

    h = math.floor((h_w[0] + 2*pad[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + 2*pad[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    return h, w

def get_model_params(model):
    params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_dict[name] = param.data
    return params_dict

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

    def get_output_size(self,h_w):
        return convtransp_output_shape(h_w, kernel_size=(1,self.kernelSize), stride=1, pad=(0,self.time_padding), dilation=1)

# https://gist.github.com/bdsaglam/b16de6ae6662e7a783e06e58e2c5185a
class DepthwiseConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, depth_multiplier=1, kernel_size=3, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        out_channels = in_channels * depth_multiplier
        super().__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=in_channels,bias=bias,
            padding_mode=padding_mode)


# https://github.com/LIKANblk/AML_EEG_challenge/blob/master/src/model_torch.py
class MI_EEGNet_stem(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,Chans=64, Samples=80,
           dropoutRates=(0.25,0.25), kernLength1=16,kernLength2=16, kernLength3=5, poolKern1=4,poolKern2=8, F1=64,
           D=4, norm_rate=0.25, dropoutType='Dropout'):
        super(MI_EEGNet_stem,self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        self.output_sizes = {}
        #block1

        time_padding = int((kernLength1//2))
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=F1,kernel_size =(1,kernLength1),padding=(0,time_padding), stride=1,bias=False)
        self.output_sizes['conv1']=convtransp_output_shape((Chans,Samples), kernel_size=(1,kernLength1), stride=1,  pad=(0,time_padding))

        self.batchnorm1 = nn.BatchNorm2d(num_features=F1, affine=True)

        self.depthwise1 = nn.Conv2d(in_channels=F1,out_channels=F1*D,kernel_size=(Chans,1),groups=F1,padding=0,bias=False)
        self.output_sizes['depthwise1'] = convtransp_output_shape(self.output_sizes['conv1'], kernel_size=(Chans,1),stride=1, pad=0)

        self.batchnorm2 = nn.BatchNorm2d(num_features=F1*D, affine=True)
        self.activation_block1 = nn.ELU()

        self.avg_pool_block1 = nn.AdaptiveAvgPool2d((1,int(self.output_sizes['depthwise1'][1]/2)))
        self.output_sizes['avg_pool_block1'] = (1,int(self.output_sizes['depthwise1'][1]/2))

        self.dropout_block1 = nn.Dropout(p=dropoutRates[0])
        
    def forward(self,input):
        out_dims = {}
        #input = batch size x 1 x channel x signal length
        block1 = self.conv1(input)
        out_dims['conv1'] = block1.size()
        block1 = self.batchnorm1(block1)
        block1 = self.depthwise1(block1)
        out_dims['depthwise1'] = block1.size()
        #print('depthwise size',block1.size())
        block1 = self.batchnorm2(block1)
        block1 = self.activation_block1(block1)
        block1 = self.avg_pool_block1(block1)
        out_dims['avg_pool_block1'] = block1.size()
        #print( 'avgpool dim',out_dims['avg_pool_block1'])
        block1 = self.dropout_block1(block1)
        return block1, out_dims


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
        self.output_sizes = {}

        time_padding = int((kernLength1//2))
        self.conv1 = nn.Conv2d(in_channels=Chans,out_channels=F,kernel_size =(1,kernLength1),padding=(0,time_padding), stride=1,bias=False)
        self.output_sizes['conv1']=convtransp_output_shape((Chans,Samples), kernel_size=(1,kernLength1), stride=1,
                                                           pad=(0,time_padding))
        
        self.separable_block1 = SeparableConv2d(in_channels=F, out_channels=F,depth=1, kernel_size=kernLength2)
        self.output_sizes['separable_block1'] = self.separable_block1.get_output_size(self.output_sizes['conv1'])

        self.batchnorm1 = nn.BatchNorm2d(num_features=F, affine=True)
        self.activation_block1 = nn.ELU()
        
        self.dropout_block1 = nn.Dropout(p=dropoutRates)

        self.separable_block2 = SeparableConv2d(in_channels=F, out_channels=F,depth=1, kernel_size=kernLength2)
        self.output_sizes['separable_block2'] = self.separable_block2.get_output_size(self.output_sizes['separable_block1'])

        self.avg_pool_block1 = nn.AdaptiveAvgPool2d((1,int(self.output_sizes['separable_block2'][1]/2)))
        self.output_sizes['avg_pool_block1'] = (1,int(self.output_sizes['separable_block2'][1]/2))

    def forward(self,input):
        out_dims = {}
        #print('stem input shape',input.shape)
        block1 = self.conv1(input)
        out_dims['conv1'] = block1.size()
        block1 = self.separable_block1(block1)
        out_dims['separable_block1'] = block1.size()
        block1 = self.batchnorm1(block1)
        block1 = self.activation_block1(block1)
        block1 = self.dropout_block1(block1)
        block1 = self.separable_block2(block1)
        out_dims['separable_block2'] = block1.size()
        block1 = self.avg_pool_block1(block1)
        out_dims['avg_pool_block1'] = block1.size()
        return block1, out_dims

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
        self.conv3_1 = nn.Conv2d(in_channels=Chans,out_channels=64,kernel_size =(1,1), stride=1,bias=False)

        self.conv4_1 = nn.Conv2d(in_channels=Chans,out_channels=64,kernel_size =(1,1), stride=(1,2),bias=False) #padding=(0,0),

    def forward(self, input):
        out_dims = {}
        #print('inception input shape',input.shape)        
        block1 = self.branch1(input)[0]
        block2 = self.branch2(input)[0]
        block3 = self.conv3_1(self.avgpool1(input))
        block4 = self.conv4_1(input)
        #print(block1.shape, block2.shape, block3.shape, block4.shape)
        concat = torch.cat([block1, block2, block3, block4], axis=1)
        return concat


class MI_EEGNet_predict(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,nb_classes, Chans=256, Samples=20,
           dropoutRates=0.25, kernLength1=5,F=256, D=4, norm_rate=0.25, dropoutType='Dropout'):
        super(MI_EEGNet_predict,self).__init__()
        self.Chans = Chans
        self.Samples = Samples
    
        self.batchnorm1 = nn.BatchNorm2d(num_features=Chans, affine=True)
        self.activation_block1 = nn.ELU()

        self.separable_block = SeparableConv2d(in_channels=Chans, out_channels=F ,depth=1, kernel_size=kernLength1)

        self.batchnorm2 = nn.BatchNorm2d(num_features=F, affine=True)
        self.activation_block2 = nn.ELU()

        self.avg_pool_block = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = nn.Dropout(dropoutRates)

        n_size = self.get_features_dim(Chans,Samples)
        self.dense = nn.Linear(n_size,nb_classes)


    def get_features_dim(self,Chans,Samples):
        bs = 1
        x = Variable(torch.rand((bs,Chans,1, Samples)))
        output_feat = self.forward_features(x)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward_features(self,input):

        block1 = self.batchnorm1(input)
        block1 = self.activation_block1(block1)
        #print(block1.shape)
        block1 = self.separable_block(block1)

        block2 = self.batchnorm2(block1)
        block2 = self.activation_block2(block2)

        block2 = self.dropout(block2)

        block2 = self.avg_pool_block(block2)
        return block2

    def forward(self, input):
        features = self.forward_features(input)
        batch_size = features.shape[0]
        flatten_feats = features.reshape(batch_size,-1)
        out = self.dense(flatten_feats)
        return out, flatten_feats

class MI_EEGNet(nn.Module):
    '''Data shape = (trials, kernels, channels, samples), which for the
        input layer, will be (trials, 1, channels, samples).'''
    #TODO resolve problems with avg padding when the end of the epoch lost
    #TODO possible solution via padding or AdaptiveAvgPool2d
    def __init__(self,nb_classes, Chans=64, Samples=80):
        super(MI_EEGNet,self).__init__()
        self.Chans = Chans
        self.Samples = Samples
    
        self.stem = MI_EEGNet_stem(Chans=Chans, Samples=Samples)
        self.inception = Inception(Chans=256, Samples=Samples//2)
        self.prediction = MI_EEGNet_predict(nb_classes=nb_classes, Chans=256, Samples=Samples//4)

    def forward(self, input, embedding=False):
        out,_ = self.stem(input) 
        out = self.inception(out)
        out, feature = self.prediction(out)
        if embedding==True:
            return out, feature
        else: 
            return out

    def weights_init(self):
        def weights_init(m):
             if isinstance(m, nn.Conv2d):
                 torch.nn.init.xavier_uniform(m.weight.data)
             if isinstance(m,nn.Linear):
                 torch.nn.init.xavier_uniform(m.weight.data)
                 m.bias.data.fill_(0.01) 


#############################EEGChannelNet ################################

class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=padding, dilation=dilation, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spatial_layers, stride, input_height):
        super().__init__()

        kernel_list = []
        for i in range(num_spatial_layers):
            kernel_list.append(((input_height // (i + 1))))

        padding = []
        for kernel in kernel_list:
            temp_pad = math.floor((kernel - 1) / 2)  # - 1 * (kernel[1] // 2 - 1)
            padding.append((temp_pad))

        feature_height = input_height // stride

        self.layers = nn.ModuleList([
            nn.Conv1d(
                in_channels, out_channels, kernel_list[i], stride, padding[i], 1
            ) for i in range(num_spatial_layers)
        ])

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)

        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, kernel_size, stride, dilation_list, in_size):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (n_layers - len(dilation_list))

        padding = []
        # Compute padding for each temporal layer to have a fixed size output
        # Output size is controlled by striding to be 1 / 'striding' of the original size
        for dilation in dilation_list:
            filter_size = kernel_size * dilation - 1
            temp_pad = math.floor((filter_size - 1) / 2) - 1 * (dilation // 2 - 1)
            padding.append((temp_pad))

        self.layers = nn.ModuleList([
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding[i], dilation_list[i]
            ) for i in range(n_layers)
        ])

    def forward(self, x):
        features = []
        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)
        return out

class FeaturesExtractor(nn.Module):
    def __init__(self, in_channels, temp_channels, out_channels, input_width, in_height,
                 temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                 num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride):
        super().__init__()

        self.temporal_block = TemporalBlock(
            in_channels, temp_channels, num_temporal_layers, temporal_kernel, temporal_stride, temporal_dilation_list, input_width
        )

        self.spatial_block = SpatialBlock(
            temp_channels * num_temporal_layers, out_channels, num_spatial_layers, spatial_stride, in_height
        )

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers
                ),
                ConvLayer2D(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers, down_kernel, down_stride, 0, 1
                )
            ) for i in range(num_residual_blocks)
        ])

        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers, out_channels, down_kernel, 1, 0, 1
        )

    def forward(self, x):
        out = self.temporal_block(x)

        out = self.spatial_block(out)

        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)

        out = self.final_conv(out)
        
        return out

class EEGChannelNet(nn.Module):

    def __init__(self, in_channels=1, temp_channels=10, out_channels=50, num_classes=2, embedding_size=1000,
                 input_width=80, input_height=64, temporal_dilation_list=[(1,1),(1,2),(1,4),(1,8),(1,16)],
                 temporal_kernel=(1,33), temporal_stride=(1,2),
                 num_temp_layers=4,
                 num_spatial_layers=4, spatial_stride=(2,1), num_residual_blocks=4, down_kernel=3, down_stride=2):
        super().__init__()

        self.encoder = FeaturesExtractor(in_channels, temp_channels, out_channels, input_width, input_height,
                                     temporal_kernel, temporal_stride,
                                     temporal_dilation_list, num_temp_layers,
                                     num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride
                                     )

        encoding_size = self.encoder(torch.zeros(1, in_channels, input_height, input_width)).contiguous().view(-1).size()[0]

        self.classifier = nn.Sequential(
            nn.Linear(encoding_size, embedding_size),
            nn.ReLU(True),
            nn.Linear(embedding_size, num_classes), 
        )

    def forward(self, x):
        out = self.encoder(x)
        print(out.shape)
        out = out.view(x.size(0), -1)

        out = self.classifier(out)

        return out



if __name__=="__main__":
	batch_size = 5
	chans =22
	#net = EEGNet_fusion(4,Chans=chans, Samples=1000)
	net = MI_EEGNet(4,Chans=chans, Samples=1000)
	from torchinfo import summary
	summary(net.cuda(), (batch_size, 1,chans, 1000))

'''
batch_size = 10
chans =64
net = EEGNet_v2(4,Chans=chans, Samples=80)
print(net.forward(Variable(torch.Tensor(np.random.rand(batch_size, 1, chans, 80)))))    #batchx1xdatapointsxchannels
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())
'''
'''

net = EEGNet_m(4)
print(net.forward(Variable(torch.Tensor(np.random.rand(10, 1, 64, 600)))))    #batchx1xdatapointsxchannels
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

'''


