import time

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from cifar10_set.Cifar10_Converter import Cifar10Converter


class Cifar10Running:
    def __init__(self, model, optimizer, criterion, batch_size, max_epoch):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', self.device)
        self.model = model.to(self.device)
        cifar10conv = Cifar10Converter(batch_size=batch_size)
        (self.train_batches, self.test_batches, self.class_labels) = cifar10conv.run_cifar_converter()
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epoch = max_epoch

        self.run()

    def train(self, epoch, max_epoch):
        self.model.train()
        train_loss, train_acc = 0, 0

        with tqdm(total=len(self.train_batches), leave=False, unit_scale=True) as qbar:
            for x, t in self.train_batches:
                x, t = x.to(self.device), t.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, t)
                train_loss += loss.item()
                train_acc += (output.max(1)[1] == t).sum().item()
                loss.backward()
                self.optimizer.step()
                qbar.set_description("Training epoch {}/{} ".format(epoch + 1, max_epoch))
                qbar.update(1)

        avg_train_loss = train_loss / len(self.train_batches.dataset)
        avg_train_correct = train_acc / len(self.train_batches.dataset) * 100.

        # time.sleep(1)
        # print('(Training)\tEpoch: {}/{} | Loss: {:.4f} | Accuracy: {:.1f} %'.format((epoch + 1), max_epoch, avg_train_loss, avg_train_correct))
        # time.sleep(1)

        return avg_train_loss, avg_train_correct

    def test(self, epoch, max_epoch):
        self.model.eval()
        test_loss, test_acc = 0, 0

        with tqdm(total=len(self.test_batches), leave=False, unit_scale=True) as qbar:
            with torch.no_grad():
                for x, t in self.test_batches:
                    x, t = x.to(self.device), t.to(self.device)
                    output = self.model(x)
                    loss = self.criterion(output, t)
                    test_loss += loss.item()
                    test_acc += (output.max(1)[1] == t).sum().item()

                    qbar.set_description("Testing epoch {}/{} ".format(epoch + 1, max_epoch))
                    qbar.update(1)

        avg_test_loss = test_loss / len(self.test_batches.dataset)
        avg_test_correct = test_acc / len(self.test_batches.dataset) * 100.

        time.sleep(1)
        print('(Test)\tEpoch: {}/{} | Loss: {:.4f} | Accuracy: {:.1f} %'.format((epoch + 1), max_epoch, avg_test_loss, avg_test_correct))
        time.sleep(1)

        return avg_test_loss, avg_test_correct

    def loss_acc_show(self, max_epoch, train_loss, train_accuracy, test_loss, test_accuracy):
        x = range(max_epoch)

        img, axis = plt.subplots(1, 2, figsize=(10, 5))

        axis[0].set_title('Loss of Training and Test', fontsize=10)
        axis[0].set_xlabel('Epoch')
        axis[0].set_ylabel('Loss')
        axis[0].plot(x, train_loss, label='train')
        axis[0].plot(x, test_loss, label='test')
        axis[0].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=14)

        axis[1].set_title('Accuracy of Training and Test', fontsize=10)
        axis[1].set_xlabel('Epoch')
        axis[1].set_ylabel('Accuracy (%)')
        axis[1].plot(x, train_accuracy, label='train')
        axis[1].plot(x, test_accuracy, label='test')
        axis[1].legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1, fontsize=14)

        plt.show()

    def run(self):
        train_loss_list, train_acc_list = [], []
        test_loss_list, test_acc_list = [], []

        for epoch in range(self.max_epoch):
            train_loss, train_accuracy = self.train(epoch, self.max_epoch)
            test_loss, test_accuracy = self.test(epoch, self.max_epoch)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_accuracy)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_accuracy)

        self.loss_acc_show(self.max_epoch, train_loss_list, train_acc_list, test_loss_list, test_acc_list)
