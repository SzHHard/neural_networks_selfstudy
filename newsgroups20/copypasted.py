from sklearn.datasets import fetch_20newsgroups
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import string

from nn_utils_module.test_nn import test_nn
from nn_utils_module.train_nn import train_nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

print('item: ', train_data.data[0][:150])  # Пример текста: "I was wondering if anyone out there..."
print('label: ', train_data.target[0])      # Метка: 7 (класс 'rec.autos')

numeric_labels = train_data.target
class_names = test_data.target_names
text_labels = [class_names[i] for i in numeric_labels]
list(set(text_labels))

print('number of label kinds: ', len(class_names), class_names,)


import torch.nn as nn
import torch.optim as optim

# Параметры
VOCAB_SIZE = 10000
EMBED_DIM = 64
HIDDEN_DIM = 128
MAX_LEN = 100  # Максимальная длина текста
BATCH_SIZE = 16
EPOCHS = 30


# 1. Создание словаря
def build_vocab(texts, vocab_size):
    counter = Counter()
    for text in texts:
        tokens = text.lower().translate(
            str.maketrans('', '', string.punctuation)
        ).split()
        counter.update(tokens)
    vocab = {'<pad>': 0, '<unk>': 1}
    vocab.update({word: i + 2 for i, (word, _) in enumerate(counter.most_common(vocab_size - 2))})
    return vocab


vocab = build_vocab(train_data.data, VOCAB_SIZE)


# 2. Датасет для PyTorch
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.lower().translate(
            str.maketrans('', '', string.punctuation)
        ).split()[:self.max_len]
        indices = [self.vocab.get(token, 1) for token in tokens]  # 1 = <unk>
        indices = indices + [0] * (self.max_len - len(indices))  # Паддинг
        return torch.tensor(indices), torch.tensor(self.labels[idx])


# 3. Создание DataLoader
train_dataset = NewsDataset(train_data.data, train_data.target, vocab, MAX_LEN)
test_dataset = NewsDataset(test_data.data, test_data.target, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# 4. Простая модель (без RNN!)
class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        pooled = embedded.mean(dim=1)  # Усреднение по словам (Bag-of-Words)
        return self.fc(pooled)


# 5. Обучение
model = BagOfWordsClassifier(len(vocab), EMBED_DIM, len(train_data.target_names))
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())



for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")
    train_nn(model, train_loader, loss_fn, optimizer, isMute=True)
    loss = test_nn(model, test_loader, loss_fn)
