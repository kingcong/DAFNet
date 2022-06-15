from utils import *
from modules import *


class SpatialAttention(nn.Cell):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=0, has_bias=False) # infer a one-channel attention map
        self.concat = ops.Concat(axis=1)
        self.mean = ops.ReduceMean(keep_dims=True)

    def construct(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = self.mean(ftr, 1) # [B, 1, H, W], average
        ftr_max = self.mean(ftr, 1) # [B, 1, H, W], average
        # ftr_max, _ = ops.ArgMaxWithValue(axis=1, keep_dims=True)(ftr) # [B, 1, H, W], max
        ftr_cat = self.concat([ftr_avg, ftr_max]) # [B, 2, H, W]
        att_map = nn.Sigmoid()(self.conv(ftr_cat)) # [B, 1, H, W]
        return att_map
    
    
class CPA(nn.Cell):
    # Cascaded Pyramid Attention
    def __init__(self, in_channels):
        super(CPA, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.conv_1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        self.SA0 = SpatialAttention()
        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.concat = ops.Concat(axis=1)
        self.DS2 = ops.AvgPool(kernel_size=2, strides=2)
        self.DS4 = ops.AvgPool(kernel_size=4, strides=4)

    def construct(self, ftr):
        # ftr: [B, C, H, W]
        d0 = self.conv_0(ftr) # [B, C/2, H, W]
        d1 = self.conv_1(self.DS2(ftr)) # [B, C/4, H/2, W/2]
        d2 = self.conv_2(self.DS4(ftr)) # [B, C/4, H/4, W/4]
        # level-2
        a2 = self.SA2(d2) #  [B, 1, H/4, W/4]
        d2 = a2*d2 + d2 # [B, C/4, H/4, W/4]
        # level-1
        d1 = self.concat([d1, US2(d2)]) # [B, C/2, H/2, W/2]
        a1 = self.SA1(d1) # [B, 1, H/2, W/2]
        d1 = a1*d1 + d1 # [B, C/2, H/2, W/2]
        # level-0
        d0 = self.concat([d0, US2(d1)]) # [B, C, H, W]
        a0 = self.SA0(d0) # [B, 1, H, W]
        return a0, d0


class ChannelRecalibration(nn.Cell):
    def __init__(self, in_channels):
        super(ChannelRecalibration, self).__init__()
        inter_channels = in_channels // 4 # channel squeezing
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.avg_fc = nn.SequentialCell(nn.Dense(in_channels=in_channels, out_channels=inter_channels, has_bias=False),
                           nn.ReLU(),
                           nn.Dense(in_channels=inter_channels, out_channels=in_channels, has_bias=False))
        self.max_pool = ops.ReduceMax(keep_dims=True)
        self.max_fc = nn.SequentialCell(nn.Dense(in_channels=in_channels, out_channels=inter_channels, has_bias=False),
                           nn.ReLU(),
                           nn.Dense(in_channels=inter_channels, out_channels=in_channels, has_bias=False))
        self.unsqueeze = ops.ExpandDims()

    def construct(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = self.avg_fc(self.avg_pool(ftr, (2, 3)).squeeze(-1).squeeze(-1)) # [B, C]
        ftr_max = self.max_fc(self.max_pool(ftr,(2, 3)).squeeze(-1).squeeze(-1)) # [B, C]
        weights = self.unsqueeze(self.unsqueeze(nn.Sigmoid()(ftr_avg + ftr_max), -1), -1) # [B, C, 1, 1]
        out = weights * ftr
        return out
    
    
class GFA(nn.Cell):
    # Global Feature Aggregation
    def __init__(self, in_channels, squeeze_ratio=4):
        super(GFA, self).__init__()
        inter_channels = in_channels // squeeze_ratio # reduce computation load
        self.conv_q = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.delta = mindspore.Parameter(mindspore.Tensor([0.1], dtype=mindspore.float32)) # initiate as 0.1
        self.cr = ChannelRecalibration(in_channels)
        self.softmax = ops.Softmax(axis=1)
        self.bmm = ops.BatchMatMul(transpose_a=True)

    def construct(self, ftr):
        B, C, H, W = ftr.shape
        P = H * W
        ftr_q = self.conv_q(ftr).view(B, -1, P)
        ftr_k = self.conv_k(ftr).view(B, -1, P) # [B, C', P]
        ftr_v = self.conv_v(ftr).view(B, -1, P) # [B, C, P]
        weights = self.bmm(ftr_q, ftr_k) # column-wise softmax, [B, P, P]
        weights = self.softmax(weights) # column-wise softmax, [B, P, P]
        G = ops.BatchMatMul()(ftr_v, weights).view(B, C, H, W)
        out = self.delta * G + ftr
        out_cr = self.cr(out)
        return out_cr
    
    
class GCA(nn.Cell):
    # Global Context-aware Attention
    def __init__(self, in_channels, use_pyramid):
        super(GCA, self).__init__()
        assert isinstance(use_pyramid, bool)
        self.use_pyramid = use_pyramid
        self.gfa = GFA(in_channels)
        if self.use_pyramid:
            self.cpa = CPA(in_channels)
        else:
            self.sau = SpatialAttention()

    def construct(self, ftr):
        ftr_global = self.gfa(ftr)
        if self.use_pyramid:
            att, ftr_refined = self.cpa(ftr_global)
            return att, ftr_refined
        else:
            att = self.sau(ftr_global)
            return att, ftr_global
        
        
class AttentionFusion(nn.Cell):
    def __init__(self, num_att_maps):
        super(AttentionFusion, self).__init__()
        dim = 256
        self.conv_1 = ConvBlock(num_att_maps, dim, 3, False, 'ReLU')
        self.conv_2 = ConvBlock(dim, dim, 3, False, 'ReLU')
        self.conv_3 = ConvBlock(dim, 1, 3, False, 'Sigmoid')

    def construct(self, concat_att_maps):
        fusion_att_maps = self.conv_3(self.conv_2(self.conv_1(concat_att_maps)))
        return fusion_att_maps  
    
    