from dataset import *
from network import *
from mindspore import context

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")

# modelatrs
import moxing as mox
parser = argparse.ArgumentParser(description='dafnet-MS train.')
parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
args, unknown = parser.parse_known_args()

mox.file.copy_parallel(src_url=args.data_url + '/EORSSDzip', dst_url='/cache/data_path/EORSSD')
mox.file.copy_parallel(src_url=args.data_url + '/Checkpoints', dst_url='/cache/data_path/Checkpoints')

zip_command = "unzip -o %s -d %s" % ('/cache/data_path/EORSSD/'+ 'EORSSD.zip', '/cache/data_path/EORSSD')
os.system(zip_command)

ckpt_folder = r'/cache/data_path/Checkpoints/'
dataset_root = r'/cache/data_path/EORSSD/'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'

context.set_context(mode=context.PYNATIVE_MODE)

# output_folder = r'./Outputs/pred/DAFNet/EORSSD/Test'
# ckpt_folder = r'../Checkpoints/'
# dataset_root = r'../EORSSD/'

batch_size = 8
init_lr = 6e-4  # 1e-5
min_lr = 1e-6
train_epoch = 200
model_name = 'trained_V0186_'
train_set = EORSSD(dataset_root, 'train', aug=True)

train_loader = ds.GeneratorDataset(train_set, ["image", "label", "edge"], shuffle=True)

dt_mean, dt_std = dataset_info('EORSSD')
flip_rot = random_aug_transform()
image_transformation = [py_vision.ToTensor(), py_vision.Normalize(dt_mean, dt_std)]
label_transformation = [py_vision.ToTensor()]
edge_transformation = [py_vision.ToTensor()]

# train_loader = train_loader.map(operations=flip_rot, input_columns=["image"])
train_loader = train_loader.map(operations=image_transformation, input_columns=["image"])
train_loader = train_loader.map(operations=label_transformation, input_columns=["label"])
train_loader = train_loader.map(operations=edge_transformation, input_columns=["edge"])

train_loader = train_loader.batch(batch_size=batch_size)


class DoubleLoss(nn.Cell):
    def __init__(self):
        super(DoubleLoss, self).__init__()
        self.bce = nn.BCELoss()

    def construct(self, sm, se, label, edge):
        mask_loss = self.bce(sm, label)
        edge_loss = self.bce(se, edge)
        total_loss = 0.7 * mask_loss + 0.3 * edge_loss
        return total_loss


class NetWithLoss(nn.Cell):
    def __init__(self, net, loss):
        super(NetWithLoss, self).__init__()
        self.net = net
        self.loss = loss
        self.cast = P.Cast()

    def construct(self, image, label, edge):
        M, E = self.net(image)
        total_loss = self.loss(M, E, label, edge)
        return self.cast(total_loss, mstype.float32)


class MyTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True):
        super(MyTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()
        self.enable_clip_grad = enable_clip_grad

    def construct(self, image, label, edge):
        weights = self.weights
        loss = self.network(image, label, edge)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(image, label, edge, sens,)
        # if self.enable_clip_grad:
        #     grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss.mean()


net = DAFNet(init_path=os.path.join(ckpt_folder, 'warehouse'))
loss = DoubleLoss()
net_with_criterion = NetWithLoss(net, loss)
optimizer = nn.Adam(net.trainable_params(), learning_rate=init_lr, weight_decay=5e-4)
train_net = MyTrainOneStepCell(net_with_criterion, optimizer)
train_net.set_train()

# 加载预训练参数
# mindspore.load_param_into_net(net, mindspore.load_checkpoint(os.path.join(ckpt_folder, 'trained', 'trained_V0166_33.ckpt')))

#学习率动态变化
# data_length = train_loader.get_dataset_size()
# milestone = [x for x in range(100 * data_length, (train_epoch + 1) * data_length, 100 * data_length)]
# learning_rates = [init_lr / (10 ** x) for x in range(0, len(milestone))]
# lr = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)



for epc in range(1, train_epoch + 1):
    epoch_loss = 0
    i = 0
    for data in train_loader.create_dict_iterator():
        # prepare input data
        image, label, edge = data["image"], data["label"], data["edge"]
        B = image.shape[0]
        train_loss = train_net(image, label, edge)
        train_loss = train_loss.asnumpy()
        epoch_loss += train_loss
        i += batch_size
        if i % 200 == 0:
            print('progress: {}/1400 || train loss: {}'
                  .format(i, train_loss))
    # update learning rate
    #############################scheduler.step()
    # cache model parameters
    ckpt_path = os.path.join(ckpt_folder, 'trained', ('trained_' + str(epc) + '.ckpt'))
    if epc % 5 == 0:
        cache_model(net, ckpt_path)
        mox.file.copy(ckpt_path, 's3://daf-net/DAFNet/Checkpoints/trained/' + model_name + str(epc) + '.ckpt')
    print('########################################################################################################')
    print('{} || epoch: {} || total loss: {}'
          .format(datetime.datetime.now(), epc, epoch_loss / i * batch_size))
    print('finish training.')