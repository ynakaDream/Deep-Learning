import codecs
import os
import pickle

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class Cifar10Converter:
    def __init__(self, batch_size, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.batch_size = batch_size
        self.is_download = False
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.train_set = None
        self.test_set = None
        self.train_batches = None
        self.test_batches = None
        self.data_dir = './cifar10_set/dataset/cifar-10-batches-py/'
        self.meta_file_name = 'batches.meta'
        self.class_labels = []

    def _cifar_downloader(self):
        if not os.path.isdir(self.data_dir):
            self.is_download = True
        self.train_set = CIFAR10(root='./cifar10_set/dataset', train=True, download=self.is_download,
                                 transform=self.transform)
        self.test_set = CIFAR10(root='./cifar10_set/dataset', train=False, download=self.is_download,
                                transform=self.transform)

    def _cifar_loader(self):
        self._cifar_downloader()
        self.train_batches = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.test_batches = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def _unpickle(self, file):
        with open(file, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        return dict

    def _cifar_class_label(self):
        meta = self._unpickle(self.data_dir + self.meta_file_name)
        class_byte = meta[b'label_names']

        # Convert byte into str
        for x in codecs.iterdecode(class_byte, 'utf-8'):
            self.class_labels.append(x)

    def run_cifar_converter(self):
        self._cifar_loader()
        self._cifar_class_label()
        return (self.train_batches, self.test_batches, self.class_labels)
