import torch
from torch import nn, optim

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super(ResBlock, self).__init__()

        self.conv_lower = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace= True)
        )
        self.conv_upper = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.BatchNorm2d(filters)
        )

        self.relu = nn.ReLU(inplace=True)

    def forwared(self, x: torch.Tensor) -> torch.Tensor:
        path = self.conv_lower(x)
        path = self.conv_upper(path)

        return self.relu(path + x)

filters = 128
model = nn.Sequential(
    nn.Conv2d(3, filters, 5, padding=2),
    nn.BatchNorm2d(filters),
    nn.ReLU(inplace=True),

    ResBlock(filters),
    ResBlock(filters),
    ResBlock(filters),
    ResBlock(filters),
    ResBlock(filters),

    nn.Conv2d(filters, 1, 1, padding=0),
    nn.BatchNorm2d(1),
    nn.ReLU(inplace=True),

    nn.Flatten(),
    nn.Linear(32**2, 10),
    nn.LogSoftmax(dim=1)
).cuda()

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
]))

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset,batch_size= 512, shuffle=True, num_workers=0) #shuffle=True해주는 이유: train할때는 순서를 섞어주는 것이 매우 중요하다. 순서를 섞어주지 않으면 overfitting될 확률이 있다.

test_loader = DataLoader(test_dataset,batch_size= 512, shuffle=False, num_workers=0) #shuffle=False해주는 이유: shuffle때문에 같은 연산을 했음에도 불구하고 결과가 달라지는 경우가 발생할 수 있어서 항상 같은 결과가 나오도록 False로 해준다.

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criteria = nn.CrossEntropyLoss().cuda()

for epoch in range(1, 20+1):
    print('epoch {}'.format(epoch))

    correct, total = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()#gpu에 올려준다.

    preds = model(inputs)

    optimizer.zero_grad()

    loss = criteria(preds, labels)
    loss.backward()

    optimizer.step()

    correct += (preds.argmax(dim=1) ==labels).sum.item()
    total += len(labels)

print('train-acc: {:.4f} '.format(correct/total), end='')

correct, total=0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        preds = model(inputs)

        correct += (preds.argmax(dim=1) ==labels).sum.item()
        total += len(labels)
print('train-acc: {:.4f} '.format(correct/total), end='\n\n')

def gaussian(img):
    noise = img.data.new(img.size()).normal_(0, 0.01)
    return torch.clamp(img + noise, 0, 1)

noise_dataset = CIFAR10(root='./data', train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    gaussian
]))
noise_loader = DataLoader(noise_dataset, batch_size=512, shuffle=False, num_workers=0)

correct, total=0, 0
with torch.no_grad():
    for inputs, labels in noise_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        preds = model(inputs)

        correct += (preds.argmax(dim=1) ==labels).sum.item()
        total += len(labels)
print('with noise acc: {:.4f} '.format(correct/total))