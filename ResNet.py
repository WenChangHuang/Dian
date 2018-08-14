import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as Data


#获取数据集
train_loader = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(train_loader, batch_size=4,
                                          shuffle=True, num_workers=2)

test_loader = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=False, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(test_loader, batch_size=4,
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#构建神经网络
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


net = ResNet(ResidualBlock, [2, 2, 2, 2]).cuda()
#net = torch.load('ResNet.pkl')
#net = net.cuda()

#训练神经网络
optimizer = optim.SGD(net.parameters(), lr=0.016)
loss_func = nn.CrossEntropyLoss()

for epoch in range(10):
   train_loss = 0
   train_acc = 0
   train_loss1 = 0
   for i, data in enumerate(trainloader, 0):

       inputs, lables = data
       inputs, lables = Variable(inputs.cuda()), Variable(lables.cuda())

       optimizer.zero_grad()

       out = net(inputs)
       loss = loss_func(out, lables)
       train_loss += loss.item()

       pred = torch.max(out, 1)[1]
       train_acc += (pred == lables).sum().item()

       loss.backward()
       optimizer.step()

       train_loss += loss.data.item()

       if i % 100 == 99:
           print('[%d,%5d] loss:%.6f' % (epoch+1, i+1, train_loss/100))
           train_loss1 = train_loss1+train_loss
           train_loss = 0

       if i % 2000 == 1999:
           print('>>>>[%d,%5d] loss:%.6f' % (epoch + 1, i + 1, train_loss1 / 2000))
           print('>>>>[%d,%5d] acc: %.6f' % (epoch + 1, i + 1, train_acc / (2000 * lables.size(0))))
           train_acc = 0
           train_loss1 = 0
           torch.save(net, 'ResNet.pkl')


#测试

total = 0
correct = 0

for images, labels in testloader:

    images = Variable(images.cuda())
    outputs = net(images)

    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels.cuda()).sum()
    acc = float(correct) / total

print('accuracy = %.6f ' % (acc))