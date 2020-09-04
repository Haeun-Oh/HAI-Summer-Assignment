import torch
from torch import nn
from torch import tensor
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
import math

data = pd.read_csv("weight-height.csv")

X, y =  data['Height'], data['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dfxtrain = pd.DataFrame({'Height': X_train})
dfxtest = pd.DataFrame({'HEight': X_test})

X_train_tensor = torch.Tensor(dfxtrain.values)
X_test_tensor = torch.Tensor(dfxtest.values)

X_train_tensor = X_train_tensor.float()
X_test_tensor = X_test_tensor.float()

X_train_tensor = f.normalize(X_train_tensor, dim=0)
X_test_tensor = f.normalize(X_test_tensor, dim=0)

dfytrain = pd.DataFrame({'Weight': y_train})
dfytest = pd.DataFrame({'Weight': y_test})

y_train_tensor = torch.Tensor(dfytrain.values)
y_test_tensor = torch.Tensor(dfytest.values)

y_train_tensor = y_train_tensor.float()
y_test_tensor = y_test_tensor.float()

y_train_tensor = f.normalize(y_train_tensor, dim=0)
y_test_tensor = f.normalize(y_test_tensor, dim=0)

###############################################33
class HeightWeightDataset(Dataset):
    def __init__(self):
        self.x_data = X_test_tensor
        self.y_data = y_test_tensor

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_return = self.x_data[idx]
        y_return = self.y_data[idx]
        
        return x_return, y_return

#필요한 model 생성
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() #부모 class의 생성자를 불러온다.
        self.linear = torch.nn.Linear(1, 1)  # One in(height) and one out(weight)

    def forward(self, x):
        y_pred = self.linear(x) #y^
        return y_pred

model = Model()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X_train_tensor)

    # 2) Compute and print loss
    loss = criterion(y_pred,y_train_tensor)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

##################################
class SubmittedApp:
    def __init__(self):
        pass

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_tensor = model(input_tensor)
        return output_tensor

    def metric(self, inferred_tensor: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        return criterion(inferred_tensor,ground_truth)

dataset = HeightWeightDataset()
test_loader = DataLoader(dataset=dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            num_workers=0)

for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)

    output = SubmittedApp.run(SubmittedApp, inputs)
    print(SubmittedApp.metric(SubmittedApp, output, labels))