from utils import *


class VGG(nn.Cell):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features

    def construct(self, x):
        x = self.features(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(path, arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        pretrained_dict = mindspore.load_checkpoint(os.path.join(path, 'bb.ckpt'))
        model_dict = model.parameters_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        # model.load_state_dict(model_dict)
        mindspore.load_param_into_net(model, model_dict)
    return model


def vgg16_bn(path, pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(path, "vgg16_bn", "D", True, pretrained, progress, **kwargs)


def Backbone_VGG16_in3(path):
    net = vgg16_bn(path, pretrained=True, progress=True)
    C1 = [example[1] for example in list(net.cells_and_names())[2:9]]
    C2 = [example[1] for example in list(net.cells_and_names())[9:16]]
    C3 = [example[1] for example in list(net.cells_and_names())[16:26]]
    C4 = [example[1] for example in list(net.cells_and_names())[26:36]]
    C5 = [example[1] for example in list(net.cells_and_names())[36:45]]
    C1 = nn.SequentialCell(C1)
    C2 = nn.SequentialCell(C2)
    C3 = nn.SequentialCell(C3)
    C4 = nn.SequentialCell(C4)
    C5 = nn.SequentialCell(C5)
    return C1, C2, C3, C4, C5