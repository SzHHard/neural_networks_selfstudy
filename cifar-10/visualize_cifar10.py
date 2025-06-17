from torchvision import datasets, transforms, utils
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 2. Классы CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 3. Вывод 10 случайных изображений
plt.figure(figsize=(10, 5))
for i in range(10):
    # Берем случайное изображение
    idx = np.random.randint(0, len(testset))
    img, label = testset[idx]

    # Преобразуем тензор в numpy и меняем порядок осей (C, H, W) -> (H, W, C)
    img = img.numpy().transpose((1, 2, 0))

    # Создаем subplot
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(classes[label])
    plt.axis('off')

plt.tight_layout()
plt.show()