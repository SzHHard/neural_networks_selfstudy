def train_nn(model, train_loader, loss_fn, optimizer):
    number_of_batches = len(train_loader)
    model.train()
    for batch_index, (X, y) in enumerate(train_loader):
        X, y = X.to('cuda'), y.to('cuda')

        # Compute prediction error
        prediction = model(X)
        loss = loss_fn(prediction, y)

        # Backpropagation
        optimizer.zero_grad() # Nullify old gradients
        loss.backward() # back-propagate
        optimizer.step() # update weights

        if batch_index % 50 == 0:
            print(f'Batch: {batch_index}/{number_of_batches} | Loss: {loss:>7f} ')