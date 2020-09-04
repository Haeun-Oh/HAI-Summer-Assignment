import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import numpy as np
from torch import nn, optim, from_numpy
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
from torchvision import datasets, transforms

iris = load_iris()

X, y =  from_numpy(iris.data), from_numpy(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.float()
X_test = X_test.float()

y_train = y_train.long()
y_test = y_test.long()

class Iris_Test_Dataset(Dataset):
    def __init__(self):
        self.x_data = X_test
        self.y_data = y_test

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_return = self.x_data[idx]
        y_return = self.y_data[idx]
        return x_return, y_return

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 9)
        self.l3 = nn.Linear(9, 3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = f.relu(self.l1(x))
        out2 = f.relu(self.l2(out1))
        y_pred = self.l3(out2)
        return y_pred

model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    model.train()
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class SubmittedApp:
    def __init__(self):
        pass

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param input_tensor: (torch.Tensor) [batchsize, n_classes(3)]
        :return: (torch.Tensor) [batchsize, logit(3)]
        """
        output_tensor =  model.forward(input_tensor)
        return output_tensor

    def metric(self, inferred_tensor: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
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

dataset = Iris_Test_Dataset()
test_loader = DataLoader(dataset=dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            num_workers=0)

with torch.no_grad():                                                      
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        y_val = SubmittedApp.run(SubmittedApp, inputs)
        
        print(SubmittedApp.metric(SubmittedApp, y_val, labels))