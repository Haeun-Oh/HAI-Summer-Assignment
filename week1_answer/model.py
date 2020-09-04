import os
from app import SubmittedApp
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
            y_pred = self.linear(x)
            return y_pred

if __name__ == "__main__":
    data = pd.read_csv("weight-height.csv")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model()
    model = model.to(device)

    criterion = torch.nn.MSELoss(reduction= "mean")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    checker = SubmittedApp()

    EPOCH = 100000

    X, y = data['Height'], data['Weight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #tensor형태로 바꿔주기
    X_train = torch.from_numpy(data['Height'].values).unsqueeze(dim=1).float()
    y_train = torch.from_numpy(data['Weight'].values).unsqueeze(dim=1).float()
    
    for epoch in range(EPOCH +1):
        y_pred = model(X_train)

        #Loss function
        loss = criterion(y_pred, y_train)

        #값이 buffer에 덮어씌워지는 것이 아니라 누적이 된다.
        optimizer.zero_grad()

        loss.backward()

        optimizer.step() #갱신

        if(epoch % 100 ==0):
            print(f'Epoch: {epoch}|Loss:{loss.item()}') #item은 scalr값을 반환한다.

    #저장    
    path = os.path.join('./', "model.pth") 
    print("saving..")
    torch.save(model.state_dict(), path)