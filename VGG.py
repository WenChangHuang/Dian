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

#搭建神经网络


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out


net = VGG().cuda()
#net = torch.load('VGG.pkl')
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
           torch.save(net, 'VGG.pkl')


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
