import torch
import torch.nn as nn

import time

from data_loaders import get_data_loaders

from neural_networks_self_study.nn_utils_module.save_model import save_model
from neural_networks_self_study.nn_utils_module.test_nn import test_nn
from neural_networks_self_study.nn_utils_module.train_nn import train_nn

start_time = time.perf_counter()

BATCH_SIZE = 256

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

train_loader, test_loader = get_data_loaders(BATCH_SIZE)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_for_first_block=1):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride_for_first_block, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.15),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.15),
        )

        # Shortcut для изменения размерности
        self.shortcut = nn.Sequential() \
            if stride_for_first_block == 1 and in_channels == out_channels \
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_for_first_block, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        out = self.sequence(x)
        out = out + self.shortcut(x) # Residual connection
        return out

class CustomResNet(nn.Module):
    def _make_layer(self, out_channels, num_blocks, stride_for_first_block):
        layers = [ResidualBlock(self.in_channels, out_channels, stride_for_first_block)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride_for_first_block=1))

        self.in_channels = out_channels

        return nn.Sequential(*layers)

    def __init__(self, num_classes=10):
        super().__init__()
        self.start_sequence = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.in_channels = 64

        self.residual_layers_of_blocks = nn.Sequential(
            self._make_layer(out_channels=64, num_blocks=3, stride_for_first_block=1), # 64x32x32
            self._make_layer(out_channels=128, num_blocks=3, stride_for_first_block=2), # 128x16x16
            self._make_layer(out_channels=256, num_blocks=3, stride_for_first_block=2), # 256x8x8
            self._make_layer(out_channels=512, num_blocks=3, stride_for_first_block=2), # 512x4x4
         )

        self.finish_sequence = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.start_sequence(x)
        out = self.residual_layers_of_blocks(out)
        out = self.finish_sequence(out)
        return out

model = CustomResNet().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,              # начальный LR (обычно 0.1 для SGD)
    momentum=0.9,        # помогает быстрее сходиться
    weight_decay=5e-4,   # L2-регуляризация (сильнее, чем для Adam)
    nesterov=True        # ускоряет сходимость
)

epochs = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

for epoch in range(epochs):
    print(f"\n-------------------------------\nEpoch {epoch+1}\n")
    train_nn(model, train_loader, loss_fn, optimizer)
    loss = test_nn(model, test_loader, loss_fn, device)

    scheduler.step()

print("Done!")

save_model(model, "cifar10_residual_connections")

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time:.6f} секунд")
