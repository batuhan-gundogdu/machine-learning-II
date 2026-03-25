
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, capacity=0):
        super(Model, self).__init__()
        self.capacity = capacity
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if capacity > 0:
            self.fc1 = nn.Linear(32*32*1, 256)
            self.layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(capacity)])
            self.fc2 = nn.Linear(256, 10)
        else:
            self.fc1 = nn.Linear(32*32*1, 10)
        self.to(self._device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.capacity > 0: # if capacity is 0, we don't have any hidden layers
            x = F.relu(self.fc1(x))
            for layer in self.layers:
                x = F.relu(layer(x))
            x = self.fc2(x)
        else:
            x = self.fc1(x)
        return x

    def predict(self, x):
        output = self(x)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

    def train(self, dataloader, val_dataloader=None, epochs=20):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        if val_dataloader is not None:
            for epoch in range(epochs):
                train_correct = 0
                train_total = 0
                for batch_idx, (data, target) in enumerate(dataloader):
                    optimizer.zero_grad()
                    output = self(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(output.data, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()
                val_correct = 0
                val_total = 0
                for batch_idx, (data, target) in enumerate(val_dataloader):
                    output = self(data)
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                print(f"Epoch {epoch} - Training Accuracy: {100 * train_correct / train_total}% - Validation Accuracy: {100 * val_correct / val_total}%")

        else:
            for epoch in tqdm(range(epochs), desc="Training the model"):
                for batch_idx, (data, target) in enumerate(dataloader):
                    optimizer.zero_grad()
                    output = self(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

    def test(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                output = self(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print(f"On {total} images, the model achieved an accuracy of {100 * correct / total}%")
        