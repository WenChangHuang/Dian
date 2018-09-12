import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

k = 3

train_loader = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(train_loader, batch_size=4,
                                          shuffle=True, num_workers=2)

test_loader = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(test_loader, batch_size=1,
                                          shuffle=False, num_workers=2)


def train(trainloader):
    for i, (images, labels) in enumerate(trainloader):
        images = Variable(images.view(-1, 32*32*3))
        labels = Variable(labels)
        images = images.numpy()
        labels = labels.numpy()
        if i == 0:
            model = images
            y = labels
        else:
            model = np.append(model, images, axis=0)
            y = np.append(y, labels, axis=0)
        if i % 1000 == 999:
            print(i+1)
    np.save("moudle.npy", model)
    np.save("y.npy", y)
    return model, y



def L1(temp, images, list):
    x = np.abs(temp - images)
    x = np.sum(x)
    list.append(x)



def predict(list, y, k):
    the_min = [0 for i in range(10)]
    for i in range(k):
        temp = np.where(list == np.min(list))
        j = y[temp[0][0]]
        the_min[j] += 1

    j = np.where(the_min == np.max(the_min))
    return j[0][0]




def test(model, y, testloader, k):
    acc = 0
    for i, (images, labels) in enumerate(testloader):
        images = Variable(images.view(-1, 32 * 32 * 3))
        labels = Variable(labels)
        images = images.numpy()
        labels = labels.numpy()

        list = []
        for j in range(10000):
            temp = model[j, :]
            L1(temp, images, list)

        pre = predict(list, y, k)
        if pre == labels[0]:
            acc += 1
    print("acc is %f" %(acc/10000))



model = np.load("moudle.npy")
y = np.load("y.npy")

#train(trainloader)
test(model, y, testloader, k)