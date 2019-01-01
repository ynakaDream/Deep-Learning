import matplotlib.pyplot as plt
from tqdm import tqdm
from processing_mnist import mnist_load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


def mnist_data_process():
    (train_X, train_t), (test_X, test_t) = mnist_load()

    train_X = torch.Tensor(train_X)
    train_t = torch.LongTensor(train_t)
    test_X = torch.Tensor(test_X)
    test_t = torch.LongTensor(test_t)

    train_dataset = TensorDataset(train_X, train_t)
    test_dataset = TensorDataset(test_X, test_t)
    train_batch = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_batch = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_batch, test_batch, train_X.shape[1]


def training():
    loss_list = []
    count = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(MAX_EPOCH):
        with tqdm(total=len(train_batch), unit_scale=True) as qbar:
            for x, t in train_batch:
                optimizer.zero_grad()
                output = model.forward(x)
                loss = criterion(output, t)

                if count % 10 == 0:
                    loss_list.append(loss.item())

                loss.backward()
                optimizer.step()
                qbar.set_description("Processing epoch {}/{} ".format(epoch + 1, MAX_EPOCH))
                qbar.update(1)
                count += 1

    xx = ( MAX_EPOCH * len(train_batch) ) // 10 + 1
    plt.xlabel('Iteration * 10')
    plt.ylabel('Loss')
    plt.plot(range(xx), loss_list)
    plt.show()


def test():
    count = 0

    with torch.no_grad():
        for x, t in test_batch:
            output = model.forward(x)
            predict = torch.max(output, dim=1)[1]

            for p, target in zip(predict, t):
                if p == target:
                    count += 1

    data_num = len(test_batch.dataset)

    print('Accuracy rate of the test data: {:.1f} %'.format(count / data_num * 100.))


if __name__ == '__main__':
    train_batch, test_batch, input_size = mnist_data_process()
    output_size = 10
    MAX_EPOCH = 3

    model = NeuralNet(input_size, output_size)

    training()
    test()
