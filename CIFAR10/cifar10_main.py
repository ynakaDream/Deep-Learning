import torch.optim as optim

from cifar10_set.Cifar10_NN_Models import *
from cifar10_set.Cifar10_Running import Cifar10Running

BATCH_SIZE = 64
MAX_EPOCH = 2

model = FCModel1(3*32*32, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

Cifar10Running(model, optimizer, criterion, BATCH_SIZE, MAX_EPOCH)
