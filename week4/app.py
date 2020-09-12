import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 32
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

class SubmittedApp:
    def __init__(self):
        pass

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Main Run Method for scoring system
        :param input_tensor: (torch.Tensor) [batchsize, image_dim(784)]
        :return: (torch.Tensor) [batchsize, n_classes(10)]
        """
        output_tensor = model.forward(input_tensor)
        return output_tensor

    @staticmethod
    def metric(inferred_tensor: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Classification Accuracy
        example)
        inferred_tensor: [[0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0]]
        ground_truth: [2, 5]
        return: 0.5
        :param inferred_tensor: (torch.Tensor) [batch_size, n_classes(10)], inferred logits
        :param ground_truth:  (torch.LongTensor) [batch_size], ground truth labels
                                each consisting LongTensor ranging from 0 to 9
        :return: (torch.Tensor) metric 점수
        """
        inferred_tensor = torch.argmax(inferred_tensor, dim=-1)
        acc = torch.mean((inferred_tensor == ground_truth).to(torch.float), dim=-1)
        return acc

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        output = SubmittedApp.run(SubmittedApp, images)
        
        print('Test Accuracy of the model on the test images: {}'.format(SubmittedApp.metric(output.data, labels.data)))