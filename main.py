import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from StochasticFlip import StochasticFlip


class Sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class binary_activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        clip_mask = torch.logical_and(input >= -1, input <= 1).float()
        grad = 2 * clip_mask * (1 - input.abs())
        return grad_output * grad


class Net(nn.Module):
    def __init__(self, width=1000, binary=False):
        super().__init__()
        self.a = (torch.randint(0, 2, [width]) * 2 - 1).cuda()
        # self.w = nn.Parameter(torch.randint(0, 2, [3072, width]).float() * 2 - 1, requires_grad=True)
        self.w = nn.Parameter(torch.rand(3072, width) * 2 - 1, requires_grad=True)
        self.scale = np.sqrt(1.0 / width)
        self.set_binary(binary)

    def set_binary(self, binary):
        if binary:
            # self.sign = Sign()
            self.sign = binary_activation()
            self.binarize = lambda x: self.sign.apply(x)
        else:
            self.binarize = lambda x: x

    def forward(self, x):
        w = self.scale * self.binarize(self.w)
        filters = x @ w
        filters = F.relu(filters)
        pred = (filters * self.a.view(1, -1)).sum(1)
        return pred

net = Net(width=1000, binary=True)
net = net.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.0)
# optimizer = StochasticFlip(net.parameters(), lr=0.1)

# Process Data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 60000

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
for train_data, train_labels in trainloader:
    break
for test_data, test_labels in testloader:
    break


def process(data, labels):
    zero_data = data[labels == 0]
    one_data = data[labels == 1]
    zero_labels = labels[labels == 0]
    one_labels = labels[labels == 1]
    data = torch.cat([zero_data, one_data], 0)
    labels = torch.cat([zero_labels, one_labels], 0)
    data = data.view(data.shape[0], -1).cuda()
    labels = labels.cuda()
    return data, labels


train_data, train_labels = process(train_data, train_labels)
test_data, test_labels = process(test_data, test_labels)

criterion = nn.BCEWithLogitsLoss()

last_loss = -1
for iter in range(10000):
    optimizer.zero_grad()
    outputs = net(train_data)
    loss = criterion(outputs, train_labels.float())
    loss.backward()
    optimizer.step()
    acc = ((outputs > 0) == (train_labels > 0)).float().mean()
    rate = (loss/last_loss).item()
    last_loss = loss.detach()
    # print(net.w[:10])

    if iter % 100 == 0:
        print('Iter ', iter, ' loss ', loss.item(), ' acc ', acc.item(), ' converge rate ', rate)

net.set_binary(True)
outputs = net(train_data)
loss = criterion(outputs, train_labels.float())
acc = ((outputs > 0) == (train_labels > 0)).float().mean()
print('loss ', loss.item(), ' acc ', acc.item())