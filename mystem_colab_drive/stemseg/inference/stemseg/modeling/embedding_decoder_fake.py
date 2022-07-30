from stemseg.modeling.embedding_utils import add_spatiotemporal_offset, get_nb_embedding_dims, get_nb_free_dims
from stemseg.modeling.common import UpsampleTrilinear3D, AtrousPyramid3D, get_temporal_scales, get_pooling_layer_creator
from stemseg.utils.global_registry import GlobalRegistry

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_HEAD_REGISTRY = GlobalRegistry.get("EmbeddingHead")

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

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
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
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

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

@EMBEDDING_HEAD_REGISTRY.add("squeeze_expand_decoder")
class SqueezingExpandDecoder(nn.Module):
    def __init__(self, in_channels, inter_channels, embedding_size, tanh_activation,
                 seediness_output, experimental_dims, ConvType=nn.Conv3d,
                 PoolType=nn.AvgPool3d, NormType=nn.Identity):
        super().__init__()

        PoolingLayerCallbacks = get_pooling_layer_creator(PoolType)
        gate_channels = 256
        reduction_ratio = 16

        self.pool_types = ['avg','max']

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )

        self.block_32x = nn.Sequential(
            ConvType(in_channels, inter_channels[0], 3, stride=1, padding=1, dilation=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1, dilation=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1, dilation=1),
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
            UpsampleTrilinear3D(scale_factor=(t_scales[0], 2, 2), align_corners=False)
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

        #self.conv_44 = nn.Conv3d(256, 128, 1, bias=False)

        self.embedding_size = embedding_size

        n_free_dims = get_nb_free_dims(experimental_dims)
        self.variance_channels = self.embedding_size - n_free_dims

        self.embedding_dim_mode = experimental_dims
        embedding_output_size = get_nb_embedding_dims(self.embedding_dim_mode)

        self.conv_embedding = nn.Conv3d(inter_channels[-1], embedding_output_size, kernel_size=1, padding=0, bias=False)
        self.conv_variance = nn.Conv3d(inter_channels[-1], self.variance_channels, kernel_size=1, padding=0, bias=True)

        self.conv_seediness, self.seediness_channels = None, 0
        if seediness_output:
            self.conv_seediness = nn.Conv3d(inter_channels[-1], 1, kernel_size=1, padding=0, bias=False)
            self.seediness_channels = 1

        self.tanh_activation = tanh_activation
        self.register_buffer("time_scale", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):

        def CAM(x):
          channel_att_sum = None
          for pool_type in self.pool_types:
              if pool_type=='avg':
                  avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                  channel_att_raw = self.mlp( avg_pool )
              elif pool_type=='max':
                  max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                  channel_att_raw = self.mlp( max_pool )
              elif pool_type=='lp':
                  lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                  channel_att_raw = self.mlp( lp_pool )
              elif pool_type=='lse':
                  lse_pool = logsumexp_2d(x)
                  channel_att_raw = self.mlp( lse_pool )

              if channel_att_sum is None:
                  channel_att_sum = channel_att_raw
              else:
                  channel_att_sum = channel_att_sum + channel_att_raw

          scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
          x = x * scale
          return x


        """
        :param x: list of multiscale feature map tensors of shape [N, C, T, H, W]. For this implementation, there
        should be 4 features maps in increasing order of spatial dimensions
        :return: embedding map of shape [N, E, T, H, W]
        """
        assert len(x) == 4, "Expected 4 feature maps, got {}".format(len(x))

        feat_map_32x, feat_map_16x, feat_map_8x, feat_map_4x = x


        for i in range(0,8):
          y = feat_map_4x[:,:,i,:,:]
          y = torch.unsqueeze(CAM(y),2)
          if i==0:
            z = y;
          else:
             z = torch.cat((z, y),2)
        feat_map_4x = z

        for i in range(0,8):
          y = feat_map_8x[:,:,i,:,:]
          y = torch.unsqueeze(CAM(y),2)
          if i==0:
            z = y;
          else:
             z = torch.cat((z, y),2)
        feat_map_8x = z

        for i in range(0,8):
          y = feat_map_16x[:,:,i,:,:]
          y = torch.unsqueeze(CAM(y),2)
          if i==0:
            z = y;
          else:
             z = torch.cat((z, y),2)
        feat_map_16x = z

        for i in range(0,8):
          y = feat_map_32x[:,:,i,:,:]
          y = torch.unsqueeze(CAM(y),2)
          if i==0:
            z = y;
          else:
             z = torch.cat((z, y),2)
        feat_map_32x = z


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


        #x=self.conv_44(feat_map_4x)

        """
        for i in range(0,8):
          y = x[:,:,i,:,:]
          y = torch.unsqueeze(CAM(y),2)
          if i==0:
            z = y;
          else:
             z = torch.cat((z, y),2)
        x = z
        """

        embeddings = self.conv_embedding(x)
        if self.tanh_activation:
            embeddings = (embeddings * 0.25).tanh()

        embeddings = add_spatiotemporal_offset(embeddings, self.time_scale, self.embedding_dim_mode)

        variances = self.conv_variance(x)

        if self.conv_seediness is not None:
            seediness = self.conv_seediness(x).sigmoid()
            output = torch.cat((embeddings, variances, seediness), dim=1)
        else:
            output = torch.cat((embeddings, variances), dim=1)
        return output


class SqueezingExpandDilatedDecoder(nn.Module):
    def __init__(self, in_channels, inter_channels, embedding_size, tanh_activation,
                 seediness_output, experimental_dims, ConvType=nn.Conv3d,
                 PoolType=nn.AvgPool3d, NormType=nn.Identity):
        super().__init__()

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
            UpsampleTrilinear3D(scale_factor=(t_scales[0], 2, 2), align_corners=False)
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

        self.embedding_size = embedding_size
        n_free_dims = get_nb_free_dims(experimental_dims)
        self.variance_channels = self.embedding_size - n_free_dims

        self.experimental_dim_mode = experimental_dims
        embedding_output_size = get_nb_embedding_dims(self.experimental_dim_mode)

        self.conv_embedding = nn.Conv3d(inter_channels[-1], embedding_output_size, kernel_size=1, padding=0, bias=False)
        self.conv_variance = nn.Conv3d(inter_channels[-1], self.variance_channels, kernel_size=1, padding=0, bias=True)

        self.conv_seediness, self.seediness_channels = None, 0
        if seediness_output:
            self.conv_seediness = nn.Conv3d(inter_channels[-1], 1, kernel_size=1, padding=0, bias=False)
            self.seediness_channels = 1

        self.tanh_activation = tanh_activation
        self.register_buffer("time_scale", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        """
        :param x: list of multiscale feature map tensors of shape [N, C, T, H, W]. For this implementation, there
        should be 4 features maps in increasing order of spatial dimensions
        :return: embedding map of shape [N, E, T, H, W]
        """
        assert len(x) == 4

        feat_map_32x, feat_map_16x, feat_map_8x, feat_map_4x = x

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

        embeddings = self.conv_embedding(x)
        if self.tanh_activation:
            embeddings = (embeddings * 0.25).tanh()

        # embeddings = embeddings + grid.detach()
        embeddings = add_spatiotemporal_offset(embeddings, self.time_scale, self.experimental_dim_mode)

        variances = self.conv_variance(x)

        if self.conv_seediness is not None:
            seediness = self.conv_seediness(x).sigmoid()
            output = torch.cat((embeddings, variances, seediness), dim=1)
        else:
            output = torch.cat((embeddings, variances), dim=1)

        return output
