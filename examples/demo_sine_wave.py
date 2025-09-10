import math
import matplotlib.pyplot as plt
from neuralgrad import Value, MLP, RMSprop, MSE

# training data: sin(x) over [-2, 2]
xs = [Value(i * 0.1) for i in range(-20, 21)]
ys = [Value(math.sin(x.data)) for x in xs]

model = MLP(1, [10, 1])
optimizer = RMSprop(model.parameters(), lr=0.01)

# initial plot
with_preds = [model([x]).data for x in xs]
plt.scatter([x.data for x in xs], [y.data for y in ys], label="True sin(x)", color="blue")
plt.plot([x.data for x in xs], with_preds, label="Init predictions", color="red")
plt.legend()
plt.show()

# training
for epoch in range(200):
    optimizer.zero_grad()
    preds = [model([x]) for x in xs]
    loss = MSE(preds, ys).compute_loss()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {loss.data:.4f}")

# final plot
preds = [model([x]).data for x in xs]
plt.scatter([x.data for x in xs], [y.data for y in ys], color="blue")
plt.plot([x.data for x in xs], preds, color="red")
plt.show()
