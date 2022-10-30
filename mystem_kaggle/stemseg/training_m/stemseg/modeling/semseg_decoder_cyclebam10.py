
import torch
import torch.nn as nn

from stemseg.modeling.common import UpsampleTrilinear3D, AtrousPyramid3D, get_pooling_layer_creator, \
    get_temporal_scales
from stemseg.utils.global_registry import GlobalRegistry

from .cyclebam import *
import torch.nn.functional as F

SEMSEG_HEAD_REGISTRY = GlobalRegistry.get("SemsegHead")


@SEMSEG_HEAD_REGISTRY.add("squeeze_expand_decoder")
class SqueezeExpandDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, inter_channels, feature_scales, foreground_channel=False,
                 ConvType=nn.Conv3d, PoolType=nn.AvgPool3d, NormType=nn.Identity):
        super().__init__()
        self.is_3d = True

        assert tuple(feature_scales) == (4, 8, 16, 32)

        PoolingLayerCallbacks = get_pooling_layer_creator(PoolType)

        self.block_32x = nn.Sequential(
            ConvType(in_channels, inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[2](3, stride=(2, 1, 1), padding=1),
        )

        self.block_16x = nn.Sequential(
            ConvType(in_channels, inter_channels[1], 3, stride=1, padding=1),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[1], inter_channels[1], 3, stride=1, padding=1),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1](3, stride=(2, 1, 1), padding=1),
            # ResidualModuleWrapper(NonLocalBlock3DWithDownsamplingV2(inter_channels, 128, 1))
        )

        self.block_8x = nn.Sequential(
            ConvType(in_channels, inter_channels[2], 3, stride=1, padding=1),
            NormType(inter_channels[2]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),
        )

        self.block_4x = nn.Sequential(
            ConvType(in_channels, inter_channels[3], 3, stride=1, padding=1),
            NormType(inter_channels[3]),
            nn.ReLU(inplace=True)
        )

        t_scales = get_temporal_scales()

        self.tcbam = TCBAMMC(in_channels,8)
        self.mc = MC(in_channels,8)
        self.tcbamin = TCBAMIN(in_channels,8)

        # 32x -> 16x
        self.upsample_32_to_16 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[0], 2, 2), align_corners=False),
        )
        self.conv_16 = nn.Conv3d(inter_channels[0] + inter_channels[1], inter_channels[1], 1, bias=False)

        # 16x to 8x
        self.upsample_16_to_8 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[1], 2, 2), align_corners=False)
        )
        self.conv_8 = nn.Conv3d(inter_channels[1] + inter_channels[2], inter_channels[2], 1, bias=False)

        # 8x to 4x
        self.upsample_8_to_4 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[2], 2, 2), align_corners=False)
        )
        self.conv_4 = nn.Conv3d(inter_channels[2] + inter_channels[3], inter_channels[3], 1, bias=False)

        
        self.fskey_conv = nn.Conv2d(inter_channels[3],inter_channels[3]//4,3,padding='same')
        self.fsvalue_conv = nn.Conv2d(inter_channels[3],inter_channels[3]//4,3,padding='same')
        self.fckey_conv = nn.Conv2d(inter_channels[3],inter_channels[3]//4,3,padding='same')
        self.softmax_attn = nn.Softmax(dim=1)
        self.fckey_conv = nn.Conv2d(inter_channels[3],inter_channels[3]//4,3,padding='same')
        self.fA_conv  = nn.Conv2d(inter_channels[3]//4,inter_channels[3],1,padding='same')
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        out_channels = num_classes + 1 if foreground_channel else num_classes
        self.conv_out = nn.Conv3d(inter_channels[3], out_channels, kernel_size=1, padding=0, bias=False)

        self.has_foreground_channel = foreground_channel

    def forward(self, x):
        
        assert len(x) == 4, "Expected 4 feature maps, got {}".format(len(x))

        feat_map_32x, feat_map_16x, feat_map_8x, feat_map_4x = x[::-1]

        feat_map_32x = self.block_32x(feat_map_32x)

        # 32x to 16x
        x = self.upsample_32_to_16(feat_map_32x)
        feat_map_16x = self.block_16x(feat_map_16x)
        x = torch.cat((x, feat_map_16x), 1)
        x = self.conv_16(x)

        # 16x to 8x
        x = self.upsample_16_to_8(x)
        feat_map_8x = self.block_8x(feat_map_8x)
        x = torch.cat((x, feat_map_8x), 1)
        x = self.conv_8(x)

        # 8x to 4x
        x = self.upsample_8_to_4(x)
        feat_map_4x = self.block_4x(feat_map_4x)
        x = torch.cat((x, feat_map_4x), 1)
        x = self.conv_4(x)

        #temporal attention
        for i in range(0,8):
          fs = x[:,:,i,:,:]  #[1 C H W]
          fs = self.maxpool(fs)
          if i==0:
            fskey = self.fskey_conv(fs) #[1 C/4 H W]
            fsvalue = self.fsvalue_conv(fs) #[1 C/4 H W]
          else:
            fskey = torch.cat((fskey,self.fskey_conv(fs)))
            fsvalue = torch.cat((fsvalue,self.fsvalue_conv(fs)))

       
        #fsvalue  #[T C/4 H W]  
        #fskey    #[T C/4 H W]                                
        C = fskey.size(1); H = fskey.size(2); W = fskey.size(3); T= fskey.size(0);  
        fsvalue = torch.permute(fsvalue,(1,0,2,3)) #[c/4 T H W]
        fsvalue = torch.reshape(fsvalue,(C,H*W*T))
        #fsvalue = fsvalue.to(device='cuda:1')  
        #fskey = fskey.to(device='cuda:1')  

        def tempattn(fcx,fskey,fsvalue,C,T,H,W):
          fc = self.maxpool(fcx)
          fckey = self.fckey_conv(fc) #[1 C/4 H W]
          fckey = torch.permute(fckey,(1,0,2,3)) #[c/4 1 H W]
          fskeyi = torch.permute(fskey,(1,0,2,3)) #[c/4 T H W]
          fckey = torch.reshape(fckey,(C,H*W))
          fskeyi = torch.reshape(fskeyi,(C,H*W*T))
          #fckey = fckey.to(device='cuda:1')  
  
          #fckey = fckey.to(device='cuda:1')
          #fskeyi = fskeyi.to(device='cuda:1')  

          X = torch.tensordot(fskeyi, fckey, dims=([0], [0]));
          
          #X = X.to(device='cuda:1')
          fA = self.softmax_attn(X)
          fA = torch.reshape(fA,(H*W*T,H,W))
          #fsvalue = fsvalue.to(device='cuda:1')
          fA = torch.tensordot(fsvalue, fA, dims=([1], [0]));
          #fA = fA.to(device='cuda:0')
          fA = self.fA_conv(fA) #[1 C H W]
          fA = fA.unsqueeze(0)
          fA = self.upsample(fA)
          ft = (fcx*1 + fA*1.7).unsqueeze(2)
          #ft = ft.to(device='cuda:1')
          return ft
            
        for i in range(0,8):
          #print('iiiiiiiii',i)
          fc = x[:,:,i,:,:] #[1 C H W]
          #fc = fc.to(device='cuda:0')  
          ft = tempattn(fc,fskey,fsvalue,C,T,H,W)
          if i==0:
            ff = ft
          else:
            ff = torch.cat((ff,ft),dim=2)

        x = ff 
        
        return self.conv_out(x)


class SqueezeExpandDilatedDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, inter_channels, feature_scales, foreground_channel=False,
                 ConvType=nn.Conv3d, PoolType=nn.AvgPool3d, NormType=nn.Identity):
        super().__init__()

        assert tuple(feature_scales) == (4, 8, 16, 32)

        PoolingLayerCallbacks = get_pooling_layer_creator(PoolType)

        self.block_32x = nn.Sequential(
            AtrousPyramid3D(in_channels, 64, ((1, 3, 3), (1, 6, 6), (1, 9, 9)), inter_channels[0]),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),

            AtrousPyramid3D(inter_channels[0], 64, ((1, 3, 3), (1, 6, 6), (1, 9, 9)), inter_channels[0]),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),

            AtrousPyramid3D(inter_channels[0], 64, ((1, 3, 3), (1, 6, 6), (1, 9, 9)), inter_channels[0]),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[2]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        )

        self.block_16x = nn.Sequential(
            AtrousPyramid3D(in_channels, 64, ((1, 4, 4), (1, 8, 8), (1, 12, 12)), inter_channels[1]),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),

            AtrousPyramid3D(in_channels, 64, ((1, 4, 4), (1, 8, 8), (1, 12, 12)), inter_channels[1]),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        )

        self.block_8x = nn.Sequential(
            ConvType(in_channels, inter_channels[2], 3, stride=1, padding=1),
            NormType(inter_channels[2]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),
        )

        self.block_4x = nn.Sequential(
            ConvType(in_channels, inter_channels[3], 3, stride=1, padding=1),
            NormType(inter_channels[3]),
            nn.ReLU(inplace=True)
        )

        t_scales = get_temporal_scales()

        # 32x -> 16x
        self.upsample_32_to_16 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[0], 2, 2), align_corners=False),
        )
        self.conv_16 = nn.Conv3d(inter_channels[0] + inter_channels[1], inter_channels[1], 1, bias=False)

        # 16x to 8x
        self.upsample_16_to_8 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[1], 2, 2), align_corners=False)
        )
        self.conv_8 = nn.Conv3d(inter_channels[1] + inter_channels[2], inter_channels[2], 1, bias=False)

        # 8x to 4x
        self.upsample_8_to_4 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[2], 2, 2), align_corners=False)
        )
        self.conv_4 = nn.Conv3d(inter_channels[2] + inter_channels[3], inter_channels[3], 1, bias=False)

        # output layer
        out_channels = num_classes + 1 if foreground_channel else num_classes
        self.conv_out = nn.Conv3d(inter_channels[-1], out_channels, kernel_size=1, padding=0, bias=False)

        self.has_foreground_channel = foreground_channel

    def forward(self, x):
        assert len(x) == 4, "Expected 4 feature maps, got {}".format(len(x))

        feat_map_32x, feat_map_16x, feat_map_8x, feat_map_4x = x[::-1]

        #F4
        feat_map_4x = self.tcbam(feat_map_4x)
        MC_F4 = self.mc(feat_map_4x)
        
        
        def todo(z,MCIN):
            w = torch.permute(z, (0, 2, 1, 3, 4))
            MC = MCIN.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(w)
            t = w * MC
            y = torch.permute(t, (0, 2, 1, 3, 4))
            return y
            
        feat_map_8x = todo(feat_map_8x,MC_F4)
        feat_map_16x = todo(feat_map_16x,MC_F4)
        feat_map_32x = todo(feat_map_32x,MC_F4)
        
        
        feat_map_32x = self.block_32x(feat_map_32x)

        # 32x to 16x
        x = self.upsample_32_to_16(feat_map_32x)
        feat_map_16x = self.block_16x(feat_map_16x)
        x = torch.cat((x, feat_map_16x), 1)
        x = self.conv_16(x)

        # 16x to 8x
        x = self.upsample_16_to_8(x)
        feat_map_8x = self.block_8x(feat_map_8x)
        x = torch.cat((x, feat_map_8x), 1)
        x = self.conv_8(x)

        # 8x to 4x
        x = self.upsample_8_to_4(x)
        feat_map_4x = self.block_4x(feat_map_4x)
        x = torch.cat((x, feat_map_4x), 1)
        x = self.conv_4(x)
        

        return self.conv_out(x)
