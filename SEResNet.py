import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import visdom
import time
viz = visdom.Visdom()

# 获取数据
transform = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

train_dataset = dsets.CIFAR100(root='./data100', train=True,
                               transform=transform, download=False)
test_dataset = dsets.CIFAR100(root='./data100', train=False,
                              transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64, shuffle=False)


#搭建神经网络
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels, 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class SENet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(SENet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[0], 2)
        self.layer3 = self.make_layer(block, 256, layers[1], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

senet = SENet(ResidualBlock, [3, 8, 36, 3]).cuda()
#senet = torch.load('SE-ResNet.pkl')
#senet = senet.cuda()

#训练神经网络
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(senet.parameters(), lr=0.0005)
#可视化
# 开始时间
start_time = time.time()
# visdom 创建 line 窗口
line = viz.line(Y=np.arange(10), env="cifar100")
# 记录数据的一些状态
train_loss, test_acc, stp = [], [], []
i = 0
for epoch in range(18):

    for step, (images, labels) in enumerate(train_loader):
        i = (i+1)/100
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = senet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if step % 200 == 199:
            correct = 0
            total = 0
            print('Epoch:', epoch+1, '|Step:', step+1, '|train loss:%.4f' % loss.item())
            torch.save(senet,  'SE-ResNet.pkl')
            for images, labels in test_loader:
                images = Variable(images.cuda())
                outputs = senet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum()
                acc = float(correct) / total

            test_acc.append(acc)
            train_loss.append(loss.item())
            stp.append(i)
            print('test accuracy:%.4f' % acc)
viz.line(Y=np.column_stack(np.array([train_loss, test_acc])),
         win=line,
         opts=dict(legend=["train_loss", "test_acc"],
                    title="cafar100"),
        env="cifar100")
