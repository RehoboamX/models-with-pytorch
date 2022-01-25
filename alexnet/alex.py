import torch
import torch.nn as nn

class Alexnet(nn.Module):
    def __init__(self, num_classes=10):  #cifar10做十分类
        super(Alexnet, self).__init__()

        self.net = nn.Sequential(
            # layer1 输入：3*227*227 输出：96*27*27
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            #nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # layer2 输入：96*27*27 输出：256*13*13
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            #nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # layer3 输入：256*13*13 输出：384*13*13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # layer4 输入：384*13*13 输出：384*13*13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # layer5 输入：384*13*13 输出：256*6*6=9126
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        #全连接
        self.dense = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        self.init_bias() #按照论文初始化权重

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        #原始论文中2，4，5层的bias设置为1
        nn.init.constant_(self.net[3].bias, 1)
        nn.init.constant_(self.net[8].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 256*6*6)
        return self.dense(x)
