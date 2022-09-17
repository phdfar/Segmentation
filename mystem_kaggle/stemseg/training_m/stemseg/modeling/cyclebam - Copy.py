import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
              avg_pool = F.avg_pool3d( x, (x.size(2),x.size(3), x.size(4)), stride=(x.size(2),x.size(3), x.size(4)) )
              avg_pool = avg_pool.squeeze(-1)
              channel_att_raw = self.mlp( avg_pool )

            elif pool_type=='max':
                max_pool = F.max_pool3d( x, (x.size(2),x.size(3), x.size(4)), stride=(x.size(2),x.size(3), x.size(4)) )
                max_pool = max_pool.squeeze(-1)
                channel_att_raw = self.mlp( max_pool )

            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        #print('torch.sigmoid( channel_att_sum )',torch.sigmoid(channel_att_sum).shape)
        #print('scale',scale.shape)
        #print('#########')
        return x * scale,torch.sigmoid(channel_att_sum)
class ChannelGateIN(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGateIN, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x,MCIN):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
              avg_pool = F.avg_pool3d( x, (x.size(2),x.size(3), x.size(4)), stride=(x.size(2),x.size(3), x.size(4)) )
              avg_pool = avg_pool.squeeze(-1)
              channel_att_raw = self.mlp( avg_pool )

            elif pool_type=='max':
                max_pool = F.max_pool3d( x, (x.size(2),x.size(3), x.size(4)), stride=(x.size(2),x.size(3), x.size(4)) )
                max_pool = max_pool.squeeze(-1)
                channel_att_raw = self.mlp( max_pool )

            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        MCIN = MCIN.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)

        #print('scale',scale.shape)
        #print('MCIN',MCIN.shape)

        scale = scale + MCIN
        return x * scale,scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale
    
class TCBAM(nn.Module):
    def __init__(self, gate_channels,gate_temporal, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(TCBAM, self).__init__()
        self.ChannelGate1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.ChannelGate2 = ChannelGate(gate_temporal, 2, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out1,mc1 = self.ChannelGate1(x)
        x_transpose = torch.permute(x_out1, (0, 2, 1, 3, 4))
        x_out2,mc2 = self.ChannelGate2(x_transpose)
        x_transpose = torch.permute(x_out2, (0, 2, 1, 3, 4))
        x_sp = self.SpatialGate(x_transpose)
        x_out = x + x_sp
        return x_out
    
class MC(nn.Module):
    def __init__(self, gate_channels,gate_temporal, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(MC, self).__init__()
        self.ChannelGate1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.ChannelGate2 = ChannelGate(gate_temporal, 2, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x): 
        # x : [N, C, T, H, W]
        x_out1,mc1 = self.ChannelGate1(x) 
        x_transpose = torch.permute(x_out1, (0, 2, 1, 3, 4)) # [N, T, C, H, W]
        x_out2,mc2 = self.ChannelGate2(x_transpose)

        #x_transpose = torch.permute(x_out2, (0, 2, 1, 3, 4)) # [N, C, T, H, W] / mc2:[N, T, C, H, W]
        #x_sp = self.SpatialGate(x_transpose)
        #x_out = x + x_sp
        return mc2
    
class TCBAMIN(nn.Module):
    def __init__(self, gate_channels,gate_temporal, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(TCBAMIN, self).__init__()
        self.ChannelGate1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.ChannelGateIN = ChannelGateIN(gate_temporal, 2, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x,MCIN):
        x_out1,mc1 = self.ChannelGate1(x)
        x_transpose = torch.permute(x_out1, (0, 2, 1, 3, 4))
        x_out2,mc2 = self.ChannelGateIN(x_transpose,MCIN)
        x_transpose = torch.permute(x_out2, (0, 2, 1, 3, 4))
        x_sp = self.SpatialGate(x_transpose)
        x_out = x + x_sp
        return x_out
