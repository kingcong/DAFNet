from utils import *


class ConvBlock(nn.Cell):
    def __init__(self, ic, oc, ks, use_bn, nl):
        # ic: input channels
        # oc: output channels
        # ks: kernel size
        # use_bn: True or False
        # nl: type of non-linearity, 'Non' or 'ReLU' or 'Sigmoid'
        super(ConvBlock, self).__init__()
        assert ks in [1, 3, 5, 7]
        assert isinstance(use_bn, bool)
        assert nl in ['Non', 'ReLU', 'Sigmoid']
        self.use_bn = use_bn
        self.nl = nl
        if ks == 1:
            self.conv = nn.Conv2d(ic, oc, kernel_size=1, has_bias=False)
        else:
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, padding=0, has_bias=False) # (ks-1)//2
        if self.use_bn == True:
            self.bn = nn.BatchNorm2d(oc)
        if self.nl == 'ReLU':
            self.ac = nn.ReLU()
        if self.nl == 'Sigmoid':
            self.ac = nn.Sigmoid()

    def construct(self, x):
        y = self.conv(x)
        if self.use_bn == True:
            y = self.bn(y)
        if self.nl != 'Non':
            y = self.ac(y)
        return y
    
    
class DecodingUnit(nn.Cell):
    def __init__(self, deep_channels, shallow_channels, dec_channels, inter_ks, is_upsample):
        super(DecodingUnit, self).__init__()
        assert isinstance(is_upsample, bool)
        self.is_upsample = is_upsample
        self.unit_conv = ConvBlock(deep_channels+shallow_channels, dec_channels, 1, True, 'ReLU')
        conv_1 = ConvBlock(dec_channels, dec_channels//4, 3, True, 'ReLU')
        conv_2 = ConvBlock(dec_channels//4, dec_channels//4, inter_ks, True, 'ReLU')
        conv_3 = ConvBlock(dec_channels//4, dec_channels, 3, True, 'ReLU')
        self.bottle_neck = nn.SequentialCell(conv_1, conv_2, conv_3)
        self.concat = ops.Concat(axis=1)

    def construct(self, deep_ftr, shallow_ftr):
        if self.is_upsample:
            deep_ftr = US2(deep_ftr)
        concat_ftr = self.concat((deep_ftr, shallow_ftr))
        inter_ftr = self.unit_conv(concat_ftr)
        dec_ftr = self.bottle_neck(inter_ftr)
        return dec_ftr
 
    
class SalHead(nn.Cell):
    def __init__(self, in_channels, inter_ks):
        super(SalHead, self).__init__()
        self.conv_1 = ConvBlock(in_channels, in_channels//2, inter_ks, False, 'ReLU')
        self.conv_2 = ConvBlock(in_channels//2, in_channels//2, 3, False, 'ReLU')
        self.conv_3 = ConvBlock(in_channels//2, in_channels//8, 3, False, 'ReLU')
        self.conv_4 = ConvBlock(in_channels//8, 1, 1, False, 'Sigmoid')

    def construct(self, dec_ftr):
        dec_ftr_ups = US2(dec_ftr)
        outputs = self.conv_4(self.conv_3(self.conv_2(self.conv_1(dec_ftr_ups))))
        return outputs
    
    