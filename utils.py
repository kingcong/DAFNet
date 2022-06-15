from packages import *


def align_number(number, N):
    assert type(number) == int
    num_str = str(number)
    assert len(num_str) <= N
    return (N - len(num_str)) * '0' + num_str


def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y


def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed


def convert2img(x):
    return Image.fromarray(x*255).convert('L')


def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if ops.ArgMaxWithValue()(smap) <= negative_threshold:  ########################
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    smap.save(path)
    
    
def cache_model(model, path):
    mindspore.save_checkpoint(model, path)
        
        
def initiate(md, path):
    mindspore.load_param_into_net(md, mindspore.load_checkpoint(path))
        
        
def DS2(x):
    return ops.AvgPool(kernel_size=2, strides=2)(x)


def DS4(x):
    return ops.AvgPool(kernel_size=4, strides=4)(x)


def DS8(x):
    return ops.AvgPool(kernel_size=8, strides=8)(x)


def DS16(x):
    return ops.AvgPool(kernel_size=16, strides=16)(x)


def US2(x):
    _, _, H, W = x.shape
    return ops.ResizeBilinear((H*2, W*2))(x)


def US4(x):
    _, _, H, W = x.shape
    return ops.ResizeBilinear((H * 4, W * 4))(x)


def US8(x):
    _, _, H, W = x.shape
    return ops.ResizeBilinear((H * 8, W * 8))(x)


def US16(x):
    _, _, H, W = x.shape
    return ops.ResizeBilinear((H * 16, W * 16))(x)


def RC(F, A):
    return F * A + F


def clip(inputs,rho=1e-15,mu=1-1e-15):
    return inputs*(mu-rho)+rho


def BCELoss_OHEM(batch_size, pred, gt, num_keep):
    loss = ops.Zeros()(batch_size, mindspore.float32).cuda()
    for b in range(batch_size):
        loss[b] = ops.BinaryCrossEntropy()(pred[b,:,:,:], gt[b,:,:,:])
        sorted_loss, idx = torch.sort(loss, descending=True)
        keep_idx = idx[0:num_keep]  
        ohem_loss = loss[keep_idx]  
        ohem_loss = ohem_loss.sum() / num_keep
    return ohem_loss


def proc_loss(losses, num_total, prec=4):
    loss_for_print = []
    for l in losses:
        loss_for_print.append(np.around(l / num_total, prec))
    return loss_for_print


