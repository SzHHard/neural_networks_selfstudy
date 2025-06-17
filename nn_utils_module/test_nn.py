import torch


def test_nn(model, test_loader, loss_fn, device):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            prediction = model(X)
            test_loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    accuracy = 100. * correct / size
    print(f'Accuracy: {accuracy:.2f}%, Avg loss: {test_loss / num_batches}')
    return test_loss / num_batches

def test_nn_accuracy_only(model, test_loader):
    correct = 0
    size = len(test_loader.dataset)
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to('cuda'), y.to('cuda')

            prediction = model(X)
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    accuracy = 100. * correct / size
    print(f'Accuracy: {accuracy:.2f}%')