import codecs
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from unpickel import unpickle


class NeuralNet(nn.Module):
    '''
    Affine1 -> Relu1 -> Affine2 -> Relu2 -> Affine3
    '''

    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_size)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
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
    loss_list = []
    count = 0
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

    xx = (MAX_EPOCH * len(train_batches)) // 10
    plt.xlabel('Iteration * 10')
    plt.ylabel('Loss')
    plt.plot(range(xx), loss_list)
    plt.show()


def test():
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

    model = NeuralNet(32 * 32 * 3, 10)

    training()
    test()
