from modules import *
from attention import *
from backbone import *

    
class Encoder(nn.Cell):
    def __init__(self, init_path):
        super(Encoder, self).__init__()
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16] 

        # bb = mindspore.load_checkpoint(os.path.join(init_path, 'bb.ckpt'))
        # vgg.load_state_dict(bb)
        # self.C1, self.C2, self.C3, self.C4, self.C5 = bb.C1, bb.C2, bb.C3, bb.C4, bb.C5

        (
            self.C1,
            self.C2,
            self.C3,
            self.C4,
            self.C5,
        ) = Backbone_VGG16_in3(init_path)
        self.GCA2 = GCA(ch[1], True)
        self.GCA3 = GCA(ch[2], True)
        self.GCA4 = GCA(ch[3], False)
        self.GCA5 = GCA(ch[4], False)
        self.AF3 = AttentionFusion(2)
        self.AF4 = AttentionFusion(3)
        self.AF5 = AttentionFusion(4)
        self.concat = ops.Concat(axis=1)
        self.DS2 = ops.AvgPool(kernel_size=2, strides=2)
        self.DS4 = ops.AvgPool(kernel_size=4, strides=4)

    def construct(self, Im):  # [3, 224, 224]
        # stage-1
        F1 = self.C1(Im)  # [64, 112, 112]
        # stage-2
        F2 = self.C2(F1)  # [128, 56, 56]
        A2, F2 = self.GCA2(F2)  # [1, 56, 56] & [128, 56, 56]
        F2 = RC(F2, A2)  # [128, 56, 56]
        # stage-3
        F3 = self.C3(F2)  # [256, 28, 28]
        A3, F3 = self.GCA3(F3)  # [1, 28, 28] & [256, 28, 28]
        A3 = self.AF3(self.concat((A3, self.DS2(A2))))  # [1, 28, 28]
        F3 = RC(F3, A3)  # [256, 28, 28]
        # stage-4
        F4 = self.C4(F3)  # [512, 14, 14]
        A4, F4 = self.GCA4(F4)  # [1, 14, 14] & [512, 14, 14]
        A4 = self.AF4(self.concat((A4, self.DS2(A3), self.DS4(A2))))  # [1, 14, 14]
        F4 = RC(F4, A4) # [512, 14, 14]
        # stage-5
        F5 = self.C5(F4)  # [512, 14, 14]
        A5, F5 = self.GCA5(F5)  # [1, 14, 14] & [512, 14, 14]
        A5 = self.AF5(self.concat((A5, A4, self.DS2(A3), self.DS4(A2))))  # [1, 14, 14]
        F5 = RC(F5, A5)  # [512, 14, 14]
        # F1: [B, 64, 112, 112]
        # F2: [B, 128, 56, 56]
        # F3: [B, 256, 28, 28]
        # F4: [B, 512, 14, 14]
        # F5: [B, 512, 14, 14]
        return F1, F2, F3, F4, F5
                            
                            
class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16]
        dec_ch = [256, 384, 512, 768]
        self.du_5 = DecodingUnit(ch[4], ch[3], dec_ch[3], 3, False)
        self.du_4 = DecodingUnit(dec_ch[3], ch[2], dec_ch[2], 5, True)
        self.du_3 = DecodingUnit(dec_ch[2], ch[1], dec_ch[1], 5, True)
        self.du_2 = DecodingUnit(dec_ch[1], ch[0], dec_ch[0], 7, True)

    def construct(self, F1, F2, F3, F4, F5):
        D4 = self.du_5(F5, F4)            
        D3 = self.du_4(D4, F3)
        D2 = self.du_3(D3, F2)
        D1 = self.du_2(D2, F1)
        # D1: [256, 112, 112]
        # D2: [384, 56, 56]
        # D3: [512, 28, 28]
        # D4: [768, 14, 14]
        return D1, D2, D3, D4


class DAFNet(nn.Cell):
    def __init__(self, init_path):
        super(DAFNet, self).__init__()   
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16]
        dec_ch = [256, 384, 512, 768]
        self.encoder = Encoder(init_path)
        initiate(self.encoder, os.path.join(init_path, 'ei.ckpt'))
        self.decoder = Decoder()
        initiate(self.decoder, os.path.join(init_path, 'di.ckpt'))
        mh = SalHead(dec_ch[0], 7)
        eh = SalHead(dec_ch[0], 7)
        self.head = nn.CellList([mh, eh])
        self.bce = nn.BCELoss()  ###############################################

    def construct(self, image):
        # image, label, edge = data[0], data[1], data[2]
        F1, F2, F3, F4, F5 = self.encoder(image)
        D1, D2, D3, D4 = self.decoder(F1, F2, F3, F4, F5)
        sm = self.head[0](D1)
        se = self.head[1](D1)
        return sm, se

    def compute_loss(self, sm, se, label, edge):
        mask_loss = self.bce(sm, label)
        edge_loss = self.bce(se, edge)
        total_loss = 0.7 * mask_loss + 0.3 * edge_loss
        return total_loss
        
        