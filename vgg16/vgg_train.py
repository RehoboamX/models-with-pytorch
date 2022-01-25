import os
import time
import torch
import wandb
from torch import nn
from dataset import load_data_cifar10
from evaluate import Accumulator, accurate_num, evaluate_accuracy
from vgg16 import VGG16

wandb.init(project='vgg16')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

root_path = os.getcwd()
model_name = 'vgg16.pth'
model_path = os.path.join(root_path, 'model', model_name)

# num_epochs = 60
# batch_size = 32
# momentum = 0.9
# lr_decay = 0.0005
# lr_init = 0.01
# classes = 10

config = wandb.config
config.num_epochs = 60
config.batch_size = 32
config.momentum = 0.9
config.lr_decay = 0.0005
config.lr_init = 0.01
config.classes = 10

vgg16 = VGG16(num_classes=config.classes).to(device)
train_iter, val_iter = load_data_cifar10(config.batch_size)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(params=vgg16.parameters(), lr=config.lr_init, momentum=config.momentum, weight_decay=config.lr_decay)


def train_epoch(net, train_iter, loss, updater):
    metric = Accumulator(3)
    for imgs, classes in train_iter:
        imgs, classes = imgs.to(device), classes.to(device)
        classes_hat = net(imgs)
        l = loss(classes_hat, classes)
        updater.zero_grad()
        l.sum().backward()
        updater.step()
        with torch.no_grad():
            metric.add(float(l.sum()), accurate_num(classes_hat, classes), classes.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, val_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        val_acc = evaluate_accuracy(net, val_iter)
        wandb.log({
            "train loss": train_loss,
            "train acc": train_acc,
            "val acc": val_acc
        })
        print('save model')
        state = {'model': vgg16.state_dict(), 'optimizer': trainer.state_dict(),
                 'epoch': epoch + 1}
        torch.save(state, model_path)
       # wandb.save(model_path)
        print(f'epoch{epoch + 1}:  train loss: {train_loss:.5f}  train acc: {train_acc:.2%}  val acc: {val_acc:.2%}')


start = time.time()
wandb.watch(vgg16, log="all")
train(vgg16, train_iter, val_iter, loss, config.num_epochs, trainer)
end = time.time()
print('Time:{:.6f} s'.format(end - start))

