import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

print(torch.__version__)  # Версия PyTorch
print(torch.cuda.is_available())

# Конфигурация
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128

# 1. Загрузка и предобработка данных
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# Загрузка данных и построение словаря
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), max_tokens=MAX_VOCAB_SIZE, specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])


# Функция для обработки текста
def text_pipeline(text):
    return vocab(tokenizer(text))


def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(1 if label == 'pos' else 0)
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
    return torch.tensor(label_list, dtype=torch.float32), pad_sequence(text_list, padding_value=vocab['<pad>'])


# 2. Подготовка DataLoader
train_iter, test_iter = IMDB(split='train'), IMDB(split='test')
train_dataset = list(train_iter)
test_dataset = list(test_iter)

# Разделение на train/valid
train_size = int(0.8 * len(train_dataset))
train_data, valid_data = random_split(train_dataset, [train_size, len(train_dataset) - train_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)


# 3. Определение модели
class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)  # [batch_size, seq_len, emb_dim]
        pooled = embedded.mean(dim=1)  # Среднее по последовательности
        out = self.dropout(torch.relu(self.fc1(pooled)))
        out = self.fc2(out)
        return self.sigmoid(out).squeeze()


model = SimpleNN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)

# 4. Обучение модели
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for labels, texts in loader:
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for labels, texts in loader:
            predictions = model(texts)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            correct += ((predictions > 0.5).float() == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')

# 5. Оценка на тестовых данных
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')