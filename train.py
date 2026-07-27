

x, y = x.to(device), y.to(device)
# Compute prediction error
prediction = model(X)
loss = loss_fn(prediction, y)
# Backpropagation
# loss.backward()
# optimizer.step()
# optimizer.zero_grad()

#Every 100 batches, calculates the loss
if batch % 100 == 0:
    loss = loss.item()
    current = (batch + 1) * len(X)
    print(f"loss: {round(loss, 2)}  [{current}/{size}]")
