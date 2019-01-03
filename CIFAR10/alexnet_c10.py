import codecs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from unpickel import unpickle


# import matplotlib.pyplot as plt


class AlexNetModel(nn.Module):
    def __init__(self):
        super(AlexNetModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.lrn1 = nn.LocalResponseNorm(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.lrn2 = nn.LocalResponseNorm(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.fc1 = nn.Linear(256, 4096)
        self.drop1 = nn.Dropout(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(inplace=True)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.lrn1(self.conv1(x))))
        x = self.pool2(F.relu(self.lrn2(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 256)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        output = self.fc3(x)
        return output


def cifar_data_process():
    '''
    Transform cifar-10 dataset into PyTorch Tensor type

    :return: (train_batches, test_batches, class_labels)
    '''

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./dataset', train=False, download=True, transform=transform)

    train_batches = DataLoader(train_set, batch_size=100, shuffle=True)
    test_batches = DataLoader(test_set, batch_size=100, shuffle=False)

    #####################################
    #     Get class names of cifar-10   #
    #####################################
    data_dir = 'dataset/cifar-10-batches-py/'
    meta_file_name = 'batches.meta'

    meta = unpickle(data_dir + meta_file_name)
    class_byte = meta[b'label_names']

    # Convert byte into str
    class_labels = []
    for x in codecs.iterdecode(class_byte, 'utf-8'):
        class_labels.append(x)

    return (train_batches, test_batches, class_labels)


def training():
    model.train()
    loss_list = []
    count = 0
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    for epoch in range(MAX_EPOCH):

        with tqdm(total=len(train_batches), unit_scale=True) as qbar:
            for x, t in train_batches:
                optimizer.zero_grad()
                output = model.forward(x)
                loss = criterion(output, t)

                if count % 10 == 0:
                    loss_list.append(loss)

                loss.backward()
                optimizer.step()

                qbar.set_description("Training epoch {}/{} ".format(epoch + 1, MAX_EPOCH))
                qbar.update(1)
                count += 1

    # xx = (MAX_EPOCH * len(train_batches)) // 10
    # plt.xlabel('Iteration * 10')
    # plt.ylabel('Loss')
    # plt.plot(range(xx), loss_list)
    # plt.show()


def test():
    model.eval()
    count = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for x, t in test_batches:
            output = model.forward(x)
            predict = torch.max(output.data, dim=1)[1]
            c = (predict == t).squeeze()

            for p, target in zip(predict, t):
                if p == target:
                    count += 1

            for i in range(100):
                label = t[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            class_labels[i], 100 * class_correct[i] / class_total[i]))

    data_num = len(test_batches.dataset)

    print('Total accuracy of the network on the 10,000 test images: {:.1f} %'.format(count / data_num * 100.))


if __name__ == '__main__':
    MAX_EPOCH = 3

    (train_batches, test_batches, class_labels) = cifar_data_process()

    model = AlexNetModel()

    training()
    test()
