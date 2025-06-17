import torch
import torch.nn as nn

from nn_utils_module.test_nn import test_nn
import time
from data_loaders import get_data_loaders
from nn_utils_module.train_nn import train_nn
from nn_utils_module.save_model import save_model

start_time = time.perf_counter()

BATCH_SIZE = 256

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

train_loader, test_loader = get_data_loaders(BATCH_SIZE)


for X, y in train_loader:
    print(f"Shape of X [N batch, Channels, Height, Width]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

class Cifar10Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1), # [batch_size, 32, 32, 32];
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Использую stride=2 в качестве обучаемого пуллинга (субдискретизация / downsampling)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # [batch_size, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # [batch_size, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [batch, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # [batch, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # [batch, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # [batch, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # [batch_size, 512, 1, 1]
            nn.Flatten(),

            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout1d(0.05),

            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.05),


            nn.Linear(128, 10, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)


    def forward(self, x):
        logits = self.sequence(x)
        return logits

model = Cifar10Model().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

epochs = 50
for epoch in range(epochs):
    print(f"\n-------------------------------\nEpoch {epoch+1}\n")
    train_nn(model, train_loader, loss_fn, optimizer)
    loss = test_nn(model, test_loader, loss_fn, device)
    scheduler.step(loss)
print("Done!")

save_model(model, "cifar10_simple_cnn")

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time:.6f} секунд")