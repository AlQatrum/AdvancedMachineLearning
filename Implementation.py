# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:39:10 2021

@author: alexi
"""
### Libraries ###
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import sys, re

######################################################################
# Code from agtree2dot from François Fleuret
# https://fleuret.org/cgi-bin/gitweb/gitweb.cgi?p=agtree2dot.git;a=summary
# see gradient graph

class Link:
    def __init__(self, from_node, from_nb, to_node, to_nb):
        self.from_node = from_node
        self.from_nb = from_nb
        self.to_node = to_node
        self.to_nb = to_nb

class Node:
    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.max_in = -1
        self.max_out = -1

def slot(node_list, n, k, for_input):
    if for_input:
        if node_list[n].max_out > 0:
            return str(node_list[n].id) + ':input' + str(k)
        else:
            return str(node_list[n].id)
    else:
        if node_list[n].max_in > 0:
            return str(node_list[n].id) + ':output' + str(k)
        else:
            return str(node_list[n].id)

def slot_string(k, for_input):
    result = ''

    if for_input:
        label = 'input'
    else:
        label = 'output'

    if k > 0:
        if not for_input: result = ' |' + result
        result +=  ' { <' + label + '0> 0'
        for j in range(1, k + 1):
            result += " | " + '<' + label + str(j) + '> ' + str(j)
        result += " } "
        if for_input: result = result + '| '

    return result

######################################################################

def add_link(node_list, link_list, u, nu, v, nv):
    if u is not None and v is not None:
        link = Link(u, nu, v, nv)
        link_list.append(link)
        node_list[u].max_in  = max(node_list[u].max_in,  nu)
        node_list[v].max_out = max(node_list[v].max_out, nv)

######################################################################

def fill_graph_lists(u, node_labels, node_list, link_list):

    if u is not None and not u in node_list:
        node = Node(len(node_list) + 1,
                    (u in node_labels and node_labels[u]) or \
                    re.search('<class \'(.*\.|)([a-zA-Z0-9_]*)\'>', str(type(u))).group(2))
        node_list[u] = node

        if hasattr(u, 'grad_fn'):
            fill_graph_lists(u.grad_fn, node_labels, node_list, link_list)
            add_link(node_list, link_list, u, 0, u.grad_fn, 0)

        if hasattr(u, 'variable'):
            fill_graph_lists(u.variable, node_labels, node_list, link_list)
            add_link(node_list, link_list, u, 0, u.variable, 0)

        if hasattr(u, 'next_functions'):
            for i, (v, j) in enumerate(u.next_functions):
                fill_graph_lists(v, node_labels, node_list, link_list)
                add_link(node_list, link_list, u, i, v, j)

######################################################################

def print_dot(node_list, link_list, out):
    out.write('digraph{\n')

    for n in node_list:
        node = node_list[n]

        if isinstance(n, torch.autograd.Variable):
            out.write(
                '  ' + \
                str(node.id) + ' [shape=note,style=filled, fillcolor="#e0e0ff",label="' + \
                node.label + ' ' + re.search('torch\.Size\((.*)\)', str(n.data.size())).group(1) + \
                '"]\n'
            )
        else:
            out.write(
                '  ' + \
                str(node.id) + ' [shape=record,style=filled, fillcolor="#f0f0f0",label="{ ' + \
                slot_string(node.max_out, for_input = True) + \
                node.label + \
                slot_string(node.max_in, for_input = False) + \
                ' }"]\n'
            )

    for n in link_list:
        out.write('  ' + \
                  slot(node_list, n.from_node, n.from_nb, for_input = False) + \
                  ' -> ' + \
                  slot(node_list, n.to_node, n.to_nb, for_input = True) + \
                  '\n')

    out.write('}\n')

######################################################################

def save_dot(x, node_labels = {}, out = sys.stdout):
    node_list, link_list = {}, []
    fill_graph_lists(x, node_labels, node_list, link_list)
    print_dot(node_list, link_list, out)

######################################################################
# End of agtree2dot

### Creating objects ###

# Small CNN to obtain Tc (color matrix transfomer)
class ColorTransformer(nn.Module):
    def __init__(self):
        super(ColorTransformer, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, stride = 1, padding = 0)
        #self.Conv1.weight.data.fill_(0)
        self.MaxPool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Relu1 = nn.ReLU()
        self.DropOut1 = nn.Dropout2d(p=0.5)
        self.Conv2 = nn.Conv2d(32, 32, kernel_size = 5, stride = 1, padding = 0)
        #self.Conv2.weight.data.fill_(0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.Relu2 = nn.ReLU()
        self.DropOut2 = nn.Dropout2d(p=0.5)
        self.Lin1 = nn.Linear(512, 100)
        #self.Lin1.weight.data.fill_(0) # only last layer need to have weights set to 0
        self.Relu3 = nn.ReLU()
        self.Lin2 = nn.Linear(100, 27)
        self.Lin2.weight.data.fill_(0)
        Bias = [1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.Lin2.bias = nn.Parameter(torch.tensor(Bias, dtype = torch.float64))
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Relu1(x)
        x = self.DropOut1(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)
        x = self.Relu2(x)
        x = self.DropOut2(x)
        x = x.view(-1, 512)
        x = self.Lin1(x)
        x = self.Relu3(x)
        x = self.Lin2(x)
        return x

# Small CNN to obtain Ts (spatial matrix transfomer)
class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, stride = 1, padding = 0) # output size : 32 x 26 x 26
        #self.Conv1.weight.data.fill_(0)
        self.MaxPool1 = nn.MaxPool2d(kernel_size = 2, stride = 2) # output size : 32 x 13 x 13
        self.Relu1 = nn.ReLU()
        self.DropOut1 = nn.Dropout2d(p=0.5)
        self.Conv2 = nn.Conv2d(32, 32, kernel_size = 5, stride = 1, padding = 0) # output size : 32 x 9 x 9
        #self.Conv2.weight.data.fill_(0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size = 3, stride = 2) # output size : 32 x 4 x 4
        self.Relu2 = nn.ReLU()
        self.DropOut2 = nn.Dropout2d(p=0.5)
        self.Lin1 = nn.Linear(512, 100)
        #self.Lin1.weight.data.fill_(0)
        self.Relu3 = nn.ReLU()
        self.Lin2 = nn.Linear(100, 6)
        self.Lin2.weight.data.fill_(0)
        Bias = [1, 0, 0, 0, 1, 0]
        self.Lin2.bias = nn.Parameter(torch.tensor(Bias, dtype = torch.float64))
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Relu1(x)
        x = self.DropOut1(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)
        x = self.Relu2(x)
        x = self.DropOut2(x)
        x = x.view(-1, 512)
        x = self.Lin1(x)
        x = self.Relu3(x)
        x = self.Lin2(x)
        
        return x
  
# Target or source transformers (At or As)
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.stn = SpatialTransformer()
        self.ctn = ColorTransformer()
        
    def DoStn(self,x):
        # get matrix for space transformation
        theta = self.stn(x)
        # reshape to the correct dim
        theta = theta.view(-1,2,3)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        
        return x
    
    def DoCtn(self,x):
        Dim = x.size()
        ImageSize = Dim[2]*Dim[3]
        Tc = self.ctn(x)
        Tc = Tc.view(-1,3,9) # Reshaping Tc to have a bsx3x9 tensor
        # Quadratic color vector
        XSqr = x**2
        XR, XG, XB = x.unbind(1) # Split image channel    
        XR = XR.unsqueeze(1) # Add one dimension
        XG = XG.unsqueeze(1)
        XB = XB.unsqueeze(1)
        Vec = torch.cat((x, XSqr, torch.mul(XR,XG), torch.mul(XR,XB), torch.mul(XG,XB)), dim = 1)
        Vec = Vec.permute(0,2,3,1) # Changing dimension order to make the .view() work
        
        ImageLinearised = Vec.reshape(-1, 9, 1)
        TcExpanded = torch.repeat_interleave(Tc, ImageSize,0)
        
        Res = torch.bmm(TcExpanded, ImageLinearised) # Matrix product inside batch between Tc and one pixel
        Res = Res.view(-1,Dim[2],Dim[3],3)
        Res = Res.permute(0,3,1,2)

        return Res
    
    def forward(self, x):
        
        x = self.DoCtn(x)
        x = self.DoStn(x)
        
        return x

# Forward or invert network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.G = FeatureExtractor_MNIST_SVHN()
        self.C = nn.ModuleList([LabelClassifier_MNIST_SVHN() for i in range(NbClassifiers)])
        
    def forward(self, x):
        x = self.G(x)
        self.Classification = []
        for Classifier in self.C:
            self.Classification.append(Classifier(x))
        return self.Classification

        
# MNIST as source and SVHN as target (table 7 in supplementary material) (G in paper)
class FeatureExtractor_MNIST_SVHN(nn.Module):
    def __init__(self):
        super(FeatureExtractor_MNIST_SVHN, self).__init__()
        self.InstanceNorm = nn.InstanceNorm2d(3) # do not change input size
        self.Conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5, stride = 1, padding = 2) # output size : 64 x 32 x 32
        self.BatchNorm1 = nn.BatchNorm2d(64) # same shape as input
        self.Activation1 = nn.ReLU()
        self.MaxPool1 = nn.MaxPool2d(kernel_size = 3, stride = 2) # outpur size : 64 x 15 x 15
        self.DropOut1 = nn.Dropout2d(p = 0.5)
        
        self.Conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 1, padding = 2) # output size : 64 x 15 x 15
        self.BatchNorm2 = nn.BatchNorm2d(64) # same shape as input
        self.Activation2 = nn.ReLU()
        self.MaxPool2 = nn.MaxPool2d(kernel_size = 3, stride = 2) # outpur size : 64 x 7 x 7
        self.DropOut2 = nn.Dropout2d(p = 0.5)
        
        self.Conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1, padding = 2) # output size : 64 x 7 x 7
        self.BatchNorm3 = nn.BatchNorm2d(128) # same shape as input
        self.Activation3 = nn.ReLU()
        self.DropOut3 = nn.Dropout2d(p = 0.5)
        
        self.Lin1 = nn.Linear(6272,3072)
        self.BatchNorm4 = nn.BatchNorm1d(3072) # same shape as input
        self.Activation4 = nn.ReLU()
        self.DropOut4 = nn.Dropout(p = 0.5)
        
        
    def forward(self, x):
        x = self.InstanceNorm(x)
        x = self.Conv1(x)
        x = self.BatchNorm1(x)
        x = self.Activation1(x)
        x = self.MaxPool1(x)
        x = self.DropOut1(x)
       
        x = self.Conv2(x)
        x = self.BatchNorm2(x)
        x = self.Activation2(x)
        x = self.MaxPool2(x)
        x = self.DropOut2(x)
        
        x = self.Conv3(x)
        x = self.BatchNorm3(x)
        x = self.Activation3(x)
        x = self.DropOut3(x)

        x = x.view(-1, 6272)
        x = self.Lin1(x)
        x = self.BatchNorm4(x)
        x = self.Activation4(x)
        x = self.DropOut4(x)

        return x

# MNIST as source and SVHN as target (table 8 in supplementary material) (Ck in paper)
class LabelClassifier_MNIST_SVHN(nn.Module):
    def __init__(self):
        super(LabelClassifier_MNIST_SVHN, self).__init__()
        self.Lin1 = nn.Linear(3072,2048)
        self.BatchNorm = nn.BatchNorm1d(2048)
        self.Activation = nn.ReLU()
        self.Lin2 = nn.Linear(2048,10)
        
    def forward(self, x):
        x = self.Lin1(x)
        x = self.BatchNorm(x)
        x = self.Activation(x)
        x = self.Lin2(x)
        
        return F.softmax(x) # all classifier end by a softmax cf article

# Lf in paper
def FoolingLoss(Output):
    Loss = torch.mean(Output * torch.log(Output))
    return Loss

# Ls in paper
def SupervisedLoss(Output, Target):
    OneHotTarget = F.one_hot(Target)
    Loss = -1*torch.mean(OneHotTarget*torch.log(Output))
    return Loss

# Lc in paper
def ConsensusLoss(Outputs):
    PHat = torch.mean(Outputs, 0)
    Loss = -1*torch.mean(PHat*torch.sum(Outputs))
    return Loss

# To compute the number of error in the prediction
def ComputeNbErrors(Target, TrueLabels, bs):
  nb_errors = 0
  _, predicted_classes = torch.max(Target, dim = 1)
  for k in range(bs):
      if TrueLabels[k] != predicted_classes[k]:
          nb_errors = nb_errors + 1
  return nb_errors


### Reminder: Gradient shall not propagate until the end 
### Sequential trainning method
def do_epoch(epoch_idx):
    # zip : The iterator stops when the shortest input iterable is exhausted.
    batch_len = min(len(mnist_trainset), len(svhn_trainset))
    
    for batch_idx, Data in enumerate(zip(mnist_trainset, svhn_trainset)):
        graph = batch_idx == 0 and epoch_idx == 0
        ### Data ###
        Source, Target = Data
        SourceData, SourceLabel = Source
        SourceData = SourceData.double()
    
        TargetData, TargetLabel = Target
        TargetData = TargetData.double()
    
        ## Place Data on GPU if available ##
        SourceData, SourceLabel = SourceData.to(device), SourceLabel.to(device)
        TargetData, TargetLabel = TargetData.to(device), TargetLabel.to(device)
        
        # Zero grad optimizer
        OptimizerTransformerSource.zero_grad()
        OptimizerTransformerTarget.zero_grad()
        OptimizerForwardNetworkG.zero_grad()
        OptimizerForwardNetworkC.zero_grad()
        OptimizerInverseNetworkG.zero_grad()
        OptimizerInverseNetworkC.zero_grad()
        
        # Update only C and G ==> As do not require grad
        for p in TransformerSource.parameters():
            p.requires_grad = False
        for p in TransformerTarget.parameters():
            p.requires_grad = False
        for p in ForwardNetwork.parameters():
            p.requires_grad = True
        for p in InverseNetwork.parameters():
            p.requires_grad = False
        
        # As(Xs)
        SourceTransformed = TransformerSource(SourceData)
        # C(G(As(Xs)))
        SourceClassification = ForwardNetwork(SourceTransformed)
        
        
        ForwardNetwork.zero_grad()
        LossTotal = 0
        for Idx, Classifier in enumerate(SourceClassification):
            Loss = SupervisedLoss(Classifier, SourceLabel)
            LossTotal += Loss
        
        LossTotal.backward()
        if graph:
            save_dot(LossTotal,{},open('./graph_step1.dot', 'w'))
        OptimizerForwardNetworkG.step()
        OptimizerForwardNetworkC.step()
    ### End step 1 ###
        # Update only C ==> G and As do not require grad
        # Graph is constructed in the forward pass, always set requires_grad params before evaluating the model
        for p in ForwardNetwork.G.parameters():
            p.requires_grad = False
            
        # At(Xt)
        TargetTransformed = TransformerTarget(TargetData)
        # C(G(At(Xt)))
        TargetClassification = ForwardNetwork(TargetTransformed)
        Classifiers = ForwardNetwork.C
        
        Classifiers.zero_grad()
        OptimizerForwardNetworkC.zero_grad()
        LossTotal2 = 0
        for Classifier, Prediction in zip(Classifiers, TargetClassification):
            Loss = FoolingLoss(Prediction)
            LossTotal2 += Loss
        if graph:
            save_dot(LossTotal2,{},open('./graph_step2.dot', 'w'))
        LossTotal2.backward()
        OptimizerForwardNetworkC.step()
    ### End step 2 ###
        # Update Only G and At
        ForwardNetwork.zero_grad()
        TransformerTarget.zero_grad()
        OptimizerForwardNetworkG.zero_grad()
        OptimizerTransformerTarget.zero_grad()
        for p in TransformerTarget.parameters():
            p.requires_grad = True
        for p in ForwardNetwork.G.parameters():
            p.requires_grad = True
        for p in ForwardNetwork.C.parameters():
            p.requires_grad = False
        for p in InverseNetwork.parameters():
            p.requires_grad = False
        
        # Recompute C(G(At(Xt)))
        TargetTransformed = TransformerTarget(TargetData)
        TargetClassification = ForwardNetwork(TargetTransformed)
        
        TargetClassificationStack = torch.stack(TargetClassification)
        Lc = ConsensusLoss(TargetClassificationStack)
        if graph:
            save_dot(Lc,{},open('./graph_step3.dot', 'w'))
        Lc.backward()
        OptimizerForwardNetworkG.step()
        OptimizerTransformerTarget.step()
        # Yt*
        TargetLabelEstimated = torch.mean(TargetClassificationStack, 0)
    ### End step 3 ###
        # Only update C- and G-
        for p in TransformerSource.parameters():
            p.requires_grad = False
        for p in ForwardNetwork.parameters():
            p.requires_grad = False
        for p in TransformerTarget.parameters():
            p.requires_grad = False
        for p in InverseNetwork.parameters():
            p.requires_grad = True
        # C-(G-(At(Xt)))
        TargetTransformed = TransformerTarget(TargetData)
        InverseTargetClassification = InverseNetwork(TargetTransformed)
       
        InverseNetwork.zero_grad()
        OptimizerInverseNetworkG.zero_grad()
        OptimizerInverseNetworkC.zero_grad()
        LossTotal3 = 0
        for Idx, InverseClassifier in enumerate(InverseTargetClassification):
            _, IndexTargetLabelEstimated = torch.max(TargetLabelEstimated, dim = 1)
            Loss = SupervisedLoss(InverseClassifier, IndexTargetLabelEstimated)
            LossTotal3 += Loss
        if graph:
            save_dot(LossTotal3,{},open('./graph_step4.dot', 'w'))
        LossTotal3.backward() # Update only G- and C-
        OptimizerInverseNetworkG.step()
        OptimizerInverseNetworkC.step()
    ### End step 4 ###
        # Only update C-
        InverseClassifiers = InverseNetwork.C
        for p in InverseNetwork.G.parameters():
            p.requires_grad = False
            
        # As(Xs)
        SourceTransformed = TransformerSource(SourceData)
        # C-(G-(As(Xs)))
        InverseSourceClassification = InverseNetwork(SourceTransformed)
        
        InverseClassifiers.zero_grad()
        OptimizerInverseNetworkC.zero_grad()
        LossTotal4 = 0
        for Prediction in InverseSourceClassification:
            Loss = FoolingLoss(Prediction)
            LossTotal4 += Loss
        if graph:
            save_dot(LossTotal4,{},open('./graph_step5.dot', 'w'))
        LossTotal4.backward() # Update only C-
        OptimizerInverseNetworkC.step()
    ### End step 5 ###
        # Only update G-
        for p in InverseNetwork.G.parameters():
            p.requires_grad = True
        for p in InverseNetwork.C.parameters():
            p.requires_grad = False
            
        # As(Xs)
        SourceTransformed = TransformerSource(SourceData)
        # C-(G-(As(Xs)))
        InverseSourceClassification = InverseNetwork(SourceTransformed)
        
        InverseNetwork.zero_grad()
        OptimizerInverseNetworkG.zero_grad()
        SourceClassificationStack = torch.stack(InverseSourceClassification)
        Lc = ConsensusLoss(SourceClassificationStack)
        if graph:
            save_dot(Lc,{},open('./graph_step6.dot', 'w'))
        Lc.backward()
        OptimizerInverseNetworkG.step()
    ### End step 6 ###
        # Only train As
        for p in InverseNetwork.G.parameters():
            p.requires_grad = False
        for p in TransformerSource.parameters():
            p.requires_grad = True
            
        # As(Xs)
        SourceTransformed = TransformerSource(SourceData)
        # C-(G-(As(Xs)))
        InverseSourceClassification = InverseNetwork(SourceTransformed)
        
        TransformerSource.zero_grad()
        
        LossTotal5 = 0
        for InverseClassifier in InverseSourceClassification:
            Loss = SupervisedLoss(InverseClassifier, SourceLabel)
            LossTotal5 += Loss
        if graph:
            save_dot(LossTotal5,{},open('./graph_step7.dot', 'w'))
        LossTotal5.backward() # Update only As
        OptimizerTransformerSource.step()
        
        if batch_idx % 10 == 0:
            nb_err = ComputeNbErrors(TargetLabelEstimated, TargetLabel, bs)
            print(f'Epoch : {epoch_idx} [Batch : {batch_idx}/{batch_len}, {nb_err} errors]')
    ### End step 7 ###
    
def train(n_epoch):
    for EpochIdx in range(n_epoch):
        do_epoch(EpochIdx)

def SaveModel(path):
    torch.save({"SourceTransformer": TransformerSource.state_dict(),
                "TargetTransformer": TransformerTarget.state_dict(),
                "ForwardNetwork": ForwardNetwork.state_dict(),
                "InverseNetwork": InverseNetwork.state_dict()}, path)

def LoadModel(path):
    TransformerSource = Transformer()
    TransformerTarget = Transformer()
    
    ForwardNetwork = Network()
    InverseNetwork = Network()
    
    params_dict = torch.load(path)
    TransformerSource.load_state_dict(params_dict["SourceTransformer"])
    TransformerTarget.load_state_dict(params_dict["TargetTransformer"])
    ForwardNetwork.load_state_dict(params_dict["ForwardNetwork"])
    InverseNetwork.load_state_dict(params_dict["InverseNetwork"])
    return TransformerSource, TransformerTarget, ForwardNetwork, InverseNetwork

### Main Program ###
### Parameters
learning_rate = 1e-6
NbClassifiers = 5 # Number of Ck to use (Nc in paper)
bs = 512 # Batch size

### Using GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


### Pipeline to transform MNIST images 
transform_mnist = transforms.Compose(
    [
     transforms.Grayscale(3), # RGB
    transforms.Pad(2,fill = 0), # padding to have a 32 x 32 image
    transforms.ToTensor()] # tensor conversion
)

transform_svhn = transforms.Compose(
    [transforms.ToTensor()] # tensor conversion
)


### Importing dataset 
mnist_trainset = DataLoader(datasets.MNIST('./data/mnist', download=True, train=True, transform=transform_mnist), 
                batch_size=bs, drop_last=True, shuffle=True)
mnist_testset = DataLoader(datasets.MNIST('./data/mnist', download=True, train=False, transform=transform_mnist), 
                batch_size=bs, drop_last=True, shuffle=True)

svhn_trainset = DataLoader(datasets.SVHN('./data/svhn', download=True, split = 'train', transform=transform_svhn), 
                batch_size=bs, drop_last=True, shuffle=True)
svhn_testset = DataLoader(datasets.SVHN('./data/svhn', download=True, split = 'test', transform=transform_svhn), 
                batch_size=bs, drop_last=True, shuffle=True)


### Model instanciation
TransformerSource = Transformer()
TransformerSource = TransformerSource.double().to(device)
TransformerSource.train()
TransformerTarget = Transformer()
TransformerTarget = TransformerTarget.double().to(device)
TransformerTarget.train()

ForwardNetwork = Network()
ForwardNetwork = ForwardNetwork.double().to(device)
ForwardNetwork.train()
InverseNetwork = Network()
InverseNetwork = InverseNetwork.double().to(device)
InverseNetwork.train()

### Optimizer instanciation
OptimizerTransformerSource = torch.optim.Adam(TransformerSource.parameters(),lr = 2e-6, weight_decay = 0.0005)
OptimizerTransformerTarget = torch.optim.Adam(TransformerTarget.parameters(),lr = 2e-6, weight_decay = 0.0005)
OptimizerForwardNetworkG = torch.optim.Adam(ForwardNetwork.G.parameters(),lr = 0.0002, weight_decay = 0.0005)
OptimizerForwardNetworkC = torch.optim.Adam(ForwardNetwork.C.parameters(),lr = 0.0002, weight_decay = 0.0005)
OptimizerInverseNetworkG = torch.optim.Adam(InverseNetwork.G.parameters(),lr = 0.0002, weight_decay = 0.0005)
OptimizerInverseNetworkC = torch.optim.Adam(InverseNetwork.C.parameters(),lr = 0.0002, weight_decay = 0.0005)

n_epoch = 100 # cf article
train(n_epoch)
SaveModel("Trained_model.pytorch")

# Compare Yt* with 


### Prediction ###
# TransformerSource, TransformerTarget, ForwardNetwork, InverseNetwork = LoadModel("Trained_model.pytorch")

# TransformerSource.eval()
# TransformerTarget.eval()

# ForwardNetwork.eval()
# InverseNetwork.eval()

# for batch_idx, Data in enumerate(zip(mnist_testset, svhn_testset)):
#     Source, Target = Data
#     SourceData, SourceLabel = Source
#     SourceData = SourceData.double()

#     TargetData, TargetLabel = Target
#     TargetData = TargetData.double()

#     SourceData, SourceLabel = SourceData.to(device), SourceLabel.to(device)
#     TargetData, TargetLabel = TargetData.to(device), TargetLabel.to(device)
#     SourceTransformed = TransformerSource(SourceData)
#     SourceClassification = ForwardNetwork(SourceTransformed)
# ### End step 1 ###
#     TargetTransformed = TransformerTarget(TargetData)
#     TargetClassification = ForwardNetwork(TargetTransformed)
#     Classifiers = ForwardNetwork.C
# ### End step 2 ###
#     TargetClassificationStack = torch.stack(TargetClassification)
#     TargetLabelEstimated = torch.mean(TargetClassificationStack, 0)
# ### End step 3 ###

# FalsePredictions = compute_nb_errors(TargetLabelEstimated, svhn_testset[1], bs)

# print("Total correct: "+ str(len(svhn_testset[1])-FalsePredictions))
# print("Accuracy: "+str(len(svhn_testset[1])-FalsePredictions)/len(svhn_testset[1]))


