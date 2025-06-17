import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as functional

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Диапазон [-1, 1]
])

# Загрузка данных
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader для батчей
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 1 канал (ч/б), 32 фильтра
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)         # Полносвязный слой
        self.fc2 = nn.Linear(128, 10)                 # 10 классов (цифры 0-9)

    def forward(self, x):
        x = functional.relu(self.conv1(x))        # [64, 32, 26, 26]
        x = functional.max_pool2d(x, 2)           # [64, 32, 13, 13]
        x = functional.relu(self.conv2(x))        # [64, 64, 11, 11]
        x = functional.max_pool2d(x, 2)           # [64, 64, 5, 5]
        x = x.view(-1, 64 * 5 * 5)       # "Разглаживаем" для полносвязного слоя
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)

model = CNN().cuda()  # Отправляем модель на GPU
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Цикл обучения
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()  # Данные на GPU
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')

# Запуск обучения (3 эпохи)
for epoch in range(1, 3):
    train(epoch)

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Accuracy: {accuracy:.2f}%')

test()