import time

import matplotlib.pyplot as plt
import torch
from torch import optim
from tqdm import tqdm

from Models import *
from stl10_data_processing import convert_into_batch


def train(epoch, max_epoch, train_loss_list, is_loss_show=False):
    model.train()
    count = 0
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    with tqdm(total=len(train_batches), leave=False, unit_scale=True) as qbar:
        for x, t in train_batches:
            optimizer.zero_grad()
            output = model.forward(x)
            loss = criterion(output, t)
            loss.backward()
            optimizer.step()
            qbar.set_description("Training epoch {}/{} ".format(epoch + 1, max_epoch))
            qbar.update(1)

            if count % 2 == 0:
                train_loss_list.append(loss)
            count += 1

    if epoch == max_epoch - 1 and is_loss_show:
        loss_show(train_loss_list)


def test(epoch, max_epoch):
    model.eval()
    correct = 0
    total = len(test_batches.dataset)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with tqdm(total=len(test_batches), leave=False, unit_scale=True) as qbar:
        with torch.no_grad():
            for x, t in test_batches:
                output = model.forward(x)
                predict = torch.max(output.data, dim=1)[1]

                correct += (predict == t).sum().item()
                c = (predict == t).squeeze()

                for i in range(batch_size):
                    label = t[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                for i in range(batch_size):
                    label = t[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                qbar.set_description("Testing epoch {}/{} ".format(epoch + 1, max_epoch))
                qbar.update(1)

    time.sleep(1)
    print('-' * 30)
    print('Epoch {}'.format(epoch + 1))
    print('*** Accuracy of classes (%) ***')

    for i in range(10):
        print('\t{} : {:.1f}'.format(class_labels[i].capitalize(), 100 * class_correct[i] / class_total[i]))

    print('\n*** TOTAL ACCURACY *** \n\t{:.1f} %\n'.format(correct / total * 100.))
    time.sleep(1)


def loss_show(loss):
    x = range(len(loss))
    plt.plot(x, loss)
    plt.show()


def run(max_epoch):
    train_loss_list = []
    for epoch in range(max_epoch):
        train(epoch, max_epoch, train_loss_list)
        test(epoch, max_epoch)


if __name__ == '__main__':
    BATCH_SIZE = 50
    MAX_EPOCH = 3

    (train_batches, test_batches, class_labels, batch_size) = convert_into_batch(batch_size=BATCH_SIZE)

    model = CNNModel4()

    run(max_epoch=MAX_EPOCH)
