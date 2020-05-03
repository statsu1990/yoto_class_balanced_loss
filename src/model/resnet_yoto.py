
"""
https://github.com/kuangliu/pytorch-cifar
MIT License

Copyright (c) 2017 liukuang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

"""
YOU ONLY TRAIN ONCE: LOSS-CONDITIONAL TRAINING OF DEEP NETWORKS
# https://openreview.net/pdf?id=HyxY6JHKwr
For YOTO models, we condition the last layer of each　convolutional block. 
The conditioning MLP has one hidden layer with 256 units on Shapes3D and 512 units on CIFAR-10. 
At training time we sample the β parameter from log-normal distribution on the interval [0.125, 1024.] 
for Shapes3D and on the interval [0.125, 512.] for CIFAR-10.

FiLM: Visual Reasoning with a General Conditioning Layer
# https://arxiv.org/pdf/1709.07871.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import mlp

class BasicBlockFilm(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # no affine before film affine
        self.bn2 = nn.BatchNorm2d(planes, affine=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, sigma, mu):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # film affine
        if sigma is not None and mu is not None:
            out = out * sigma.view(sigma.size()[0],-1,1,1) + mu.view(mu.size()[0],-1,1,1)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BottleneckFilm(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckFilm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        # no affine before film affine
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, affine=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, sigma=None, mu=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # film affine
        if sigma is not None and mu is not None:
            out = out * sigma.view(sigma.size()[0],-1,1,1) + mu.view(mu.size()[0],-1,1,1)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, sigma, mu

class ResNetYoto(nn.Module):
    def __init__(self, block, num_blocks, num_classes, param_sampler, use_flim=True):
        super(ResNetYoto, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        # yoto
        self.param_sampler = param_sampler
        if use_flim:
            layer_channels = [64*block.expansion, 64*block.expansion, 128*block.expansion, 128*block.expansion, 
                                   256*block.expansion, 256*block.expansion, 512*block.expansion, 512*block.expansion, ]
            self.sigma_gene = mlp.MultiheadMLP(self.param_sampler.n_param, n_outputs=layer_channels, 
                     common_hidden_neurons=(64,256), 
                     multi_head_hidden_neurons=((128,), (128,), (256,), (256,), (512,), (512,), (1024,), (1024,)),
                     dropout_rate=0.1)
            self.mu_gene = mlp.MultiheadMLP(self.param_sampler.n_param, n_outputs=layer_channels, 
                     common_hidden_neurons=(64,256), 
                     multi_head_hidden_neurons=((128,), (128,), (256,), (256,), (512,), (512,), (1024,), (1024,)),
                     dropout_rate=0.1)
        else:
            self.sigma_gene = None
            self.mu_gene = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x, params=None):
        if params is None:
            if self.training:
                params = self.param_sampler.sampling_params(x.size()[0])
            else:
                params = self.param_sampler.params

        if self.sigma_gene is not None and self.mu_gene is not None:
            normed_params = self.param_sampler.normalize_params(params)
            sigs = self.sigma_gene(normed_params)
            mus = self.mu_gene(normed_params)
        else:
            sigs = [None,]*8
            mus = [None,]*8

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.run_layer(out, self.layer1, sigs[0:2], mus[0:2])
        out = self.run_layer(out, self.layer2, sigs[2:4], mus[2:4])
        out = self.run_layer(out, self.layer3, sigs[4:6], mus[4:6])
        out = self.run_layer(out, self.layer4, sigs[6:8], mus[6:8])

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, params.data

    def run_layer(self, input, layer, sigs, mus):
        out = input
        for i, (ly, sig, mu) in enumerate(zip(layer, sigs, mus)):
            out = ly(out, sig, mu)
        return out

    def set_params(self, params):
        self.param_sampler.params.data = torch.from_numpy(np.array(params)[None,:].astype('float32')).cuda()

class ParamSampler(nn.Module):
    def __init__(self, param_ranges=((0.1, 2.0),), params=(1.0,), param_dist='log1m_uniform', same_param_in_batch=False):
        """
        Args:
            param_ranges: ((param1 low lim, param1 high lim), (param2 low lim, param2 high lim), ...)
            params: used parameter if self.training is False.
            param_dist: 'log1m_uniform', 'log_uniform', 'uniform'
            same_param_in_batch : If True, use same parameter in batch.
        """
        super(ParamSampler, self).__init__()
        self.param_dist = param_dist
        self.same_param_in_batch = same_param_in_batch
        
        self.n_param = len(param_ranges)
        self.param_ranges = nn.Parameter(torch.from_numpy(np.array(param_ranges)[None,:,:].astype('float32')), requires_grad =False)
        self.params = nn.Parameter(torch.from_numpy(np.array(params)[None,:].astype('float32')), requires_grad =False)

    def sampling_params(self, batch_size):
        prm_rng = self.__conv_func(self.param_ranges)

        if self.same_param_in_batch:
            prms = np.random.rand(1, self.param_ranges.size()[1])
        else:
            prms = np.random.rand(batch_size, self.param_ranges.size()[1])
        prms = torch.from_numpy(prms.astype('float32')).clone().cuda()
        prms = prms * (prm_rng[:,:,1] - prm_rng[:,:,0]) + prm_rng[:,:,0]
        prms = self.__inv_func(prms)
        return prms

    def normalize_params(self, params):
        prm_rng = self.__conv_func(self.param_ranges)
        prms = self.__conv_func(params)

        normed_prms = (prms - prm_rng[:,:,0]) / (prm_rng[:,:,1] - prm_rng[:,:,0]) * 2.0 - 1.0
        return normed_prms

    def set_params(self, params):
        self.params.data = torch.from_numpy(np.array(params)[None,:].astype('float32')).cuda()

    def __conv_func(self, x):
        if self.param_dist == 'log_uniform':
            return torch.log(x)
        elif self.param_dist == 'log1m_uniform':
            return torch.log(1 - x)
        elif self.param_dist == 'uniform':
            return x
        else:
            return

    def __inv_func(self, x):
        if self.param_dist == 'log_uniform':
            return torch.exp(x)
        elif self.param_dist == 'log1m_uniform':
            return 1 - torch.exp(x)
        elif self.param_dist == 'uniform':
            return x
        else:
            return

def ResNet18(num_classes=10, param_sampler=None, use_flim=True):
    return ResNetYoto(BasicBlockFilm, [2,2,2,2], num_classes, param_sampler, use_flim)

def ResNet34(num_classes=10, param_sampler=None, use_flim=True):
    return ResNetYoto(BasicBlockFilm, [3,4,6,3], num_classes, param_sampler, use_flim)

def ResNet50(num_classes=10, param_sampler=None, use_flim=True):
    return ResNetYoto(BottleneckFilm, [3,4,6,3], num_classes, param_sampler, use_flim)

def ResNet101(num_classes=10, param_sampler=None, use_flim=True):
    return ResNetYoto(BottleneckFilm, [3,4,23,3], num_classes, param_sampler, use_flim)

def ResNet152(num_classes=10, param_sampler=None, use_flim=True):
    return ResNetYoto(BottleneckFilm, [3,8,36,3], num_classes, param_sampler, use_flim)
