import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
from  datasets.base_dataset import BaseDataset

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

class Dateloader(BaseDataset):

    def __init__(self, root,mode,target_size=256, viz=False, debug=False,dataset = "imagenet"):
        super(Dateloader, self).__init__(target_size, viz, debug)
        self.mode = mode
        self.root = root
        self.dataset = dataset
        self.IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
        self.images = []
        self.gt = []

        if self.dataset == "ImageNet":
            if self.mode == "train":
                self.root = os.path.join(self.root,"train")
            else:
                self.root = os.path.join(self.root, "val")
            self.classes, self.class_to_idx = self.find_classes(self.root)
            self.images, self.gt = self.make_dataset(self.root, self.class_to_idx, extensions=self.IMG_EXTENSIONS)

        elif self.dataset == "OpenImage":
            train_list = os.listdir(os.path.join(self.root,"train","clean"))
            val_list = os.listdir(os.path.join(self.root, "val"))
            class_list = list(
                set(train_list).intersection(val_list))
            if self.mode == "train":
                self.root = os.path.join(self.root,"train","clean")
            else:
                self.root = os.path.join(self.root, "val")

            classes = [
                d for d in os.listdir(self.root) if d in class_list
            ]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(1100)}

            self.images, self.gt = self.make_dataset_openimage(self.root, class_to_idx, extensions=self.IMG_EXTENSIONS)
            print("image,gt:",len(self.images),len(self.gt))
        elif self.dataset == "CIFAR10":
            if self.mode == "train":
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (self.root, n)
                    data_dic = unpickle(dpath)
                    self.images.append(data_dic['data'])
                    self.gt = self.gt + data_dic['labels']
                self.images = np.concatenate(self.images)

                self.images = self.images.reshape((50000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
            else:
                test_dic = unpickle('%s/test_batch' % self.root)
                self.images = test_dic['data']
                self.images = self.images.reshape((10000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
                self.gt = test_dic['labels']

        elif self.dataset == "CIFAR100":
            if self.mode == "train":
                train_dic = unpickle('%s/train' % self.root)
                self.images = train_dic['data']
                self.gt = train_dic['fine_labels']
                self.images = self.images.reshape((50000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))

            else:
                test_dic = unpickle('%s/test' % self.root)
                self.images = test_dic['data']
                self.images = self.images.reshape((10000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
                self.gt = test_dic['fine_labels']

        else:
            print("improper dataset")
            raise NameError

    def make_dataset_openimage(self,root, class_to_idx, extensions):
        """Make dataset by walking all images under a root.

        Args:
            root (string): root directory of folders
            class_to_idx (dict): the map from class name to class idx
            extensions (tuple): allowed extensions

        Returns:
            images (list): a list of tuple where each element is (image, label)
        """
        images = []
        gt = []
        root = os.path.expanduser(root)
        for class_name in sorted(os.listdir(root)):
            if class_name not in class_to_idx:
                continue
            _dir = os.path.join(root, class_name)
            if not os.path.isdir(_dir):
                continue

            for _, _, fns in sorted(os.walk(_dir)):
                num = 0
                for fn in sorted(fns):
                    num = num+1
                    if has_file_allowed_extension(fn, extensions):
                        path = os.path.join(root,class_name, fn)
                        images.append(path)
                        gt.append(class_to_idx[class_name])
                    # if num>300:
                    #     break
                break
        return images,gt

    def make_dataset(self,root, class_to_idx, extensions):
        """Make dataset by walking all images under a root.

        Args:
            root (string): root directory of folders
            class_to_idx (dict): the map from class name to class idx
            extensions (tuple): allowed extensions

        Returns:
            images (list): a list of tuple where each element is (image, label)
        """
        images = []
        gt = []
        root = os.path.expanduser(root)
        for class_name in sorted(os.listdir(root)):
            _dir = os.path.join(root, class_name)
            if not os.path.isdir(_dir):
                continue

            for _, _, fns in sorted(os.walk(_dir)):
                num = 0
                for fn in sorted(fns):
                    num = num+1
                    if has_file_allowed_extension(fn, extensions):
                        path = os.path.join(root,class_name, fn)
                        images.append(path)
                        gt.append(class_to_idx[class_name])
                    # if num>300:
                    #     break
                break
        return images,gt

    def find_classes(self, root):
        """Find classes by folders under a root.

        Args:
            root (string): root directory of folders

        Returns:
            classes (list): a list of class names
            class_to_idx (dict): the map from class name to class idx
        """
        classes = [
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.images)

    # def get_imagename(self, index):
    #     return self.image[index][0]

    def load_image_gt(self, index):
        '''
        根据索引加载ground truth
        '''
        image = self.images[index]
        gt = self.gt[index]

        return image, gt

if __name__ == "__main__":
    import torch
    from torch.utils import data
    from torch import nn
    from torch.optim import lr_scheduler
    import os
    import time
    import numpy as np
    dataset = Dateloader("/data/glusterfs_cv_04/public_data/imagenet/CLS-LOC", mode="train", dataset="ImageNet")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        pin_memory=True)

    for i, (image,gt) in enumerate(data_loader):
        print(image)