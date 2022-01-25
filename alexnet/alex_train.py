import os
import time
import torch
from torch import nn
from cifar10 import load_data_cifar10
from evaluate import Accumulator, accurate_num, evaluate_accuracy
from alex import Alexnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_path = os.getcwd()
model_name = 'alexnet.pth'
model_path = os.path.join(root_path, 'model', model_name)

num_epochs = 2
batch_size = 128
momentum = 0.9
lr_decay = 0.0005
lr_init = 0.01
classes = 10

alexnet = Alexnet(num_classes=classes).to(device)
train_iter, val_iter = load_data_cifar10(batch_size)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(params=alexnet.parameters(), lr=lr_init, momentum=momentum, weight_decay=lr_decay)


def train_epoch(net, train_iter, loss, updater):
    metric = Accumulator(3)
    for imgs, classes in train_iter:
        imgs, classes = imgs.to(device), classes.to(device)
        classes_hat = net(imgs)
        l = loss(classes_hat, classes)
        updater.zero_grad()
        l.sum().backward()
        updater.step()
        metric.add(float(l.sum()), accurate_num(classes_hat, classes), classes.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, val_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        val_acc = evaluate_accuracy(net, val_iter)
        print('save model')
        state = {'model': alexnet.state_dict(), 'optimizer': trainer.state_dict(),
                 'epoch': epoch + 1}
        torch.save(state, model_path)
        print(f'epoch{epoch + 1}:  train loss: {train_loss:.5f}  train acc: {train_acc:.2%}  val acc: {val_acc:.2%}')


start = time.time()
train(alexnet, train_iter, val_iter, loss, num_epochs, trainer)
end = time.time()
print('Time:{:.6f} s'.format(end - start))

