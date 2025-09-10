from neuralgrad import MLP, Adam, CrossEntropy

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, -1.0, 1.0]
]
ys = [1, 0, 0, 1]  # binary labels

model = MLP(3, [4, 4, 1])
optimizer = Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    predictions = [model(x) for x in xs]
    loss = CrossEntropy(predictions, ys).compute_loss()
    loss.backward()
    optimizer.step()
    print(epoch, loss.data)
