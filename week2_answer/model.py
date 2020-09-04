import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os #dataset을 저장하기 위해서 불러온다.

class IrisDataset(data.Dataset):
    def __init__(self, train=True):
        super(IrisDataset, self).__init__()

        iris = load_iris()

        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

        if train:
            self.inputs, self.labels = X_train, y_train

        else:
            self.inputs, self.labels = X_test, y_test

    def __getitem__(self, index):
        inp = torch.Tensor(self.inputs[index])
        label = torch.LongTensor([self.labels[index]])
        return inp, label

    def __len__(self):
        return len(self.inputs)

#model
class IrisModel(nn.Module): #pytorch에서 제공하는 nn.Module을 상속을 받아야 model을 구현할 수 있다.
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out   

def metric(inferred_tensor: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Classification Accuracy
        example)
        inferred_tensor: [[0,0,1], [0,1,0]]
        ground_truth: [2, 0]
        return: 0.5
        :param inferred_tensor: (torch.Tensor) [batch_size, n_classes(3)], inferred logits
        :param ground_truth:  (torch.LongTensor) [batch_size], ground truth labels
                                each consisting LongTensor ranging from 0 to 2
        :return: (torch.Tensor) metric 점수
        """

        inferred_tensor = torch.argmax(inferred_tensor, dim=-1)
        acc = torch.mean((inferred_tensor == ground_truth).to(torch.float), dim=-1)
        return acc

#학습
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = IrisModel().to(device)

#dataloader구현
dataloader = data.DataLoader(IrisDataset(train=True), batch_size=32, shuffle=True, num_workers=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epoch = 100
for epoch in range(n_epoch):
    #계산을 하기 위해서 loss를 저장하는 변수(모니터링 용)
    train_loss = torch.Tensor([0]).to(device)
    train_acc = torch.Tensor([0]).to(device)

    for x, y in dataloader:
        x = x.to(device)
        y= y.squeeze(-1).to(device)

        optimizer.zero_grad()
        y_ = model(x)

        loss = criterion(y_, y)
        train_loss +=loss
        loss.backward()
        optimizer.step()

        acc = metric(y_, y)
        train_acc += acc

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    print('[{curr}/{tot}] >Loss: {loss:.2f}, Acc: {acc:.2f}%'.format(
        curr = epoch+1,
        tot = n_epoch,
        loss = train_loss.detach().cpu().item(),
        acc = float(100*train_acc.detach().cpu().item())
    ))

model.eval()
dataloader = data.DataLoader(IrisDataset(train=False), batch_size=30, shuffle = True, num_workers=0)
test_acc = torch.Tensor([0]).to(device)
for x, y in dataloader:
    x = x.to(device)
    y =y.squeeze(-1).to(device)
    y_=model(x)
    acc = metric(y_, y)
    test_acc +=acc

    test_acc /= len(dataloader)
    print(float(test_acc*100))

path = os.path.join('./', 'model.pth')
torch.save(model.state_dict(), path)

model = IrisModel().to(device)
model.load_state_dict(torch.load(path))