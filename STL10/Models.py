import torch.nn as nn
import torch.nn.functional as F


class CNNModel1(nn.Module):
    def __init__(self):
        super(CNNModel1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.fc1 = nn.Linear(20 * 92 * 92, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 20 * 92 * 92)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output


class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.fc1 = nn.Linear(20 * 92 * 92, 100)
        self.drop1 = nn.Dropout(inplace=True)
        self.fc2 = nn.Linear(100, 50)
        self.drop2 = nn.Dropout(inplace=True)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 20 * 92 * 92)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        output = self.fc3(x)
        return output


class CNNModel3(nn.Module):
    def __init__(self):
        super(CNNModel3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(20 * 96 * 96, 100)
        self.drop1 = nn.Dropout(inplace=True)
        self.fc2 = nn.Linear(100, 50)
        self.drop2 = nn.Dropout(inplace=True)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 20 * 96 * 96)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        output = self.fc3(x)
        return output


class CNNModel4(nn.Module):
    def __init__(self):
        super(CNNModel4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20 * 96 * 96, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 96 * 96)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output


class CNNModel5(nn.Module):
    def __init__(self):
        super(CNNModel5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(30)
        self.conv5 = nn.Conv2d(in_channels=30, out_channels=20, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(20)
        self.conv6 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, padding=1)
        self.batch6 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20 * 96 * 96, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.batch1(F.relu(self.conv1(x)))
        x = self.batch2(F.relu(self.conv2(x)))
        x = self.batch3(F.relu(self.conv3(x)))
        x = self.batch4(F.relu(self.conv4(x)))
        x = self.batch5(F.relu(self.conv5(x)))
        x = self.batch6(F.relu(self.conv6(x)))
        x = x.view(-1, 20 * 96 * 96)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
