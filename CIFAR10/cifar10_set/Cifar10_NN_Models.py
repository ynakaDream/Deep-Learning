import torch.nn as nn
import torch.nn.functional as F


class FCModel1(nn.Module):
    '''
        epoch: 50
        batch: 64
        nn.CrossEntropyLoss()
        optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        * Accuracy: 55.5 %
        * Loss: 0.0200
    '''
    def __init__(self, input_size, output_size):
        super(FCModel1, self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, output_size)
        self.dropout1 = nn.Dropout2d()
        self.dropout2 = nn.Dropout2d()

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        output = self.fc3(x)
        return output


class FCModel2(nn.Module):
    '''
        epoch: 50
        batch: 64
        nn.CrossEntropyLoss()
        optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        * Accuracy: 55.6 %
        * Loss: 0.0199
    '''
    def __init__(self, input_size, output_size):
        super(FCModel2, self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, output_size)
        self.dropout1 = nn.Dropout2d()
        self.dropout2 = nn.Dropout2d()
        self.dropout3 = nn.Dropout2d()
        self.dropout4 = nn.Dropout2d()
        self.dropout5 = nn.Dropout2d()

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        output = self.fc6(x)
        return output


class FCModel3(nn.Module):
    '''
        epoch: 50
        batch: 64
        nn.CrossEntropyLoss()
        optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        * Accuracy:  %
        * Loss:
    '''
    def __init__(self, input_size, output_size):
        super(FCModel3, self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.batch1 = nn.BatchNorm2d(500)
        self.fc2 = nn.Linear(500, 500)
        self.batch2 = nn.BatchNorm2d(500)
        self.fc3 = nn.Linear(500, output_size)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = self.batch1(x)
        x = F.relu(self.fc2(x))
        x = self.batch2(x)
        output = self.fc3(x)
        return output



class CNNModel1(nn.Module):
    '''
        epoch: 50
        batch: 64
        nn.CrossEntropyLoss()
        optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        * Accuracy: 55.6 %
        * Loss: 0.0270
    '''
    def __init__(self):
        super(CNNModel1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=3, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(3 * 32 * 32, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)
        self.dropout1 = nn.Dropout2d()
        self.dropout2 = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        output = self.fc3(x)
        return output


class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=3, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * 32 * 32, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.batch1(F.relu(self.conv1(x)))
        x = self.batch2(F.relu(self.conv2(x)))
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output


class CNNModel3(nn.Module):
    '''
        epoch: 50
        batch: 64
        nn.CrossEntropyLoss()
        optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        * Accuracy: 56.6 %
        * Loss: 0.0190
    '''
    def __init__(self):
        super(CNNModel3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=3, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.fc1 = nn.Linear(3 * 3 * 3, 500)
        self.drop1 = nn.Dropout2d()
        self.fc2 = nn.Linear(500, 500)
        self.drop2 = nn.Dropout2d()
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 3 * 3 * 3)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        output = self.fc3(x)
        return output


class CNNModel4(nn.Module):
    '''
        epoch: 50
        batch: 64
        nn.CrossEntropyLoss()
        optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-6)

        * Accuracy: 57.9 %
        * Loss: 0.0187

    '''
    def __init__(self):
        super(CNNModel4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=3, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.fc1 = nn.Linear(3 * 3 * 3, 500)
        self.drop1 = nn.Dropout2d()
        self.fc2 = nn.Linear(500, 500)
        self.drop2 = nn.Dropout2d()
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 3 * 3 * 3)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        output = self.fc3(x)
        return output


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
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout()
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
