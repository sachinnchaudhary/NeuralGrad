from neuralgrad import MLP
import math

# training data (XOR-like)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, -1.0, 1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])
print("Initial output:", model(xs[0]))
