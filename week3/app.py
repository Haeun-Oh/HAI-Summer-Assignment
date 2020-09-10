from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

batch_size = 64

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, 
    transform=transforms.ToTensor())

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, 
    transform=transforms.ToTensor())

train_loader = data.DataLoader(dataset=train_dataset, 
                            batch_size= batch_size, 
                            shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

class SubmittedApp:
    def __init__(self):
        pass

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        ''' Main run method for scoring system
        :param input_tensor: (torch.Tensor) [batchsize, channel(3), width, height]
        :return: (torch.Tensor) [batchsize, n_classes(10)]
        '''
        output_tensor = model.forward(input_tensor)
        return output_tensor

    def metric(self, inferred_tensor: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        ''' Calculate accuracy
        example)
        inferred_tensor: [[0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0]]
        ground_truth: [2, 5]
        return: 0.5
        because argmax inferred_tensor[0] is 2 and argmax inferred_tensor[1] is 6
        :param inferred_tensor: (torch.Tensor) [batch_size, n_classes(10)], inferred logits
        :param ground_truth: (torch.LongTensor) [batch_size], ground truth labels
                                each consisting LongTensor ranging from 0 to 9 (total 10 classes)
        :return: (torch.Tensor) metric score
        '''
        return torch.mean((inferred_tensor.argmax(dim=1) == ground_truth).float(), dim=-1)
       

for epoch in range(1, 10):
    train(epoch)
    with torch.no_grad():
        model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = SubmittedApp.run(SubmittedApp, data)
        
            print(SubmittedApp.metric(SubmittedApp, output.data, target.data))