from utils import *


def dataset_info(dt):
    assert dt in ['EORSSD']
    if dt == 'EORSSD':
        dt_mean = [0.3412, 0.3798, 0.3583]
        dt_std = [0.1148, 0.1042, 0.0990]
    return dt_mean, dt_std


def random_aug_transform():
    flip_h = py_vision.RandomHorizontalFlip(prob=1)
    flip_v = py_vision.RandomVerticalFlip(prob=1)
    angles = [0, 90, 180, 270]
    rot_angle = angles[np.random.choice(4)]
    rotate = py_vision.RandomRotation((rot_angle, rot_angle))
    r = np.random.random()
    if r <= 0.25:
        flip_rot = [flip_h, flip_v, rotate]
    elif r <= 0.5:
        flip_rot = [flip_h, rotate]
    elif r <= 0.75:
        flip_rot = [flip_v, flip_h, rotate]
    else:
        flip_rot = [flip_v, rotate]
    return flip_rot

    
class EORSSD:
    def __init__(self, root, mode, aug=False):
        self.aug = aug
        self.dt_mean, self.dt_std = dataset_info('EORSSD')
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        self.image_paths = [os.path.join(root, 'images_resized_224', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, 'labels', prefix + '.png') for prefix in self.prefixes]
        self.edge_paths = [os.path.join(root, 'edges_resized_224', prefix + '.png') for prefix in self.prefixes]
    def __getitem__(self, index):
        flip_rot = random_aug_transform()
        # image = self.image_transformation(flip_rot(Image.open(self.image_paths[index])))
        # label = self.label_transformation(flip_rot(Image.open(self.label_paths[index])))
        # edge = self.label_transformation(flip_rot(Image.open(self.edge_paths[index])))
        image = Image.open(self.image_paths[index])
        label = Image.open(self.label_paths[index]).resize((224, 224))
        edge = Image.open(self.edge_paths[index])
        return image, label, edge
    def __len__(self):
        return len(self.prefixes)
    
    