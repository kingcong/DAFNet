from dataset import *
from network import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpu_devices = list(np.arange(torch.cuda.device_count()))
multi_gpu = len(gpu_devices) > 1

output_folder = r'./Outputs/pred/DAFNet/EORSSD/Test'
ckpt_folder = r'./Checkpoints'
dataset_root = r'../Dataset/EORSSD'

batch_size = 12 * len(gpu_devices)
train_set = EORSSD(dataset_root, 'train', aug=True)
train_loader = data.DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=8, drop_last=True)

net = DAFNet(return_loss=True, init_path=os.path.join(ckpt_folder, 'warehouse')).cuda()
if multi_gpu:
    net = nn.DataParallel(net, gpu_devices)
    print('Use {} GPUs'.format(len(gpu_devices)))
else:
    print('Use a single GPU')

init_lr = 1e-5
min_lr = 1e-6
train_epoch = 24
optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epoch, eta_min=min_lr)

import os
os.system('rm -rf ./Outputs/pred/DAFNet/EORSSD/Test/*.png')

test_set = EORSSD(dataset_root, 'test', aug=False)
test_loader = data.DataLoader(test_set, shuffle=False, batch_size=1, num_workers=8, drop_last=False)


net = DAFNet(return_loss=False, init_path=os.path.join(ckpt_folder, 'warehouse')).eval().cuda()
net.load_state_dict(torch.load(os.path.join(ckpt_folder, 'trained', 'zqj_trained.pth')))


infer_time = 0
num_test = 0
for image, label, edge, prefix in test_loader:
    num_test += 1
    with torch.no_grad():
        image, label, edge = image.cuda(), label.cuda(), edge.cuda()
        B = image.size(0)
        t1 = time.time()
        smap, _ = net(image, label, edge)
        t2 = time.time()
        infer_time += (t2 - t1)
        for b in range(B):
            path = os.path.join(output_folder, prefix[b] + '.png')
            save_smap(smap[b, ...], path)
print('finish testing.')
infer_time /= num_test
print('average inference speed: {} FPS'.format(int(np.round(1/infer_time))))