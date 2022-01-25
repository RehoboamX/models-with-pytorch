import torch
from torch import nn
from Fashion_Mnist import load_data_fashion_mnist
from evaluate import Accumulator, accurate_num, evaluate_accuracy

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,512), nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 256), nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, 128), nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, 64), nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Linear(64, 32), nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Linear(32,10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.2, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.75)
train_iter, test_iter = load_data_fashion_mnist(batch_size)

def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.sum().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accurate_num(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch{epoch+1}:  train loss: {train_loss:.5f}  train acc: {train_acc:.2%}  test acc: {test_acc:.2%}')

train(net, train_iter, test_iter, loss, num_epochs, trainer)

