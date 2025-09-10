# NeuralGrad
Inspired by Andrej Karpathy's implementation of micrograd, this is the similar kind of library which can perform operations related to neural networks like fforward pass and back propagation, loss computation and perfoming gradient descent with minimal code(near 150-200 lines probably). i want to implement even more things so additionally i added importat activation functions like Relu, Sigmoid and loss function such as Cross entropy and MSE(mean squere error) with Optimization algorithm(RMSprop and Adam). The idea behind this is to get intution around how the process happens behind the hood in deep learning frameworks like Pytorch. 

### Installation 
Clone the repo:

```
bash
git clone https://github.com/<your-username>/neuralgrad.git
cd neuralgrad
pip install -r requirements.txt
```


## Example usage

Below is a simple example showing how to create values, perform operations, and compute gradients:

```
python
from neuralgrad.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

print(f'{g.data:.4f}')  ===  24.5000
g.backward()
print(f'{a.grad:.4f}')  == 140.0
print(f'{a.grad:.4f}') == 651.0
```

## Test
We tested NeuralGrad in `test.ipynb` by training a small MLP (1 → 10 → 1) on a toy regression task: fitting the function y = sin(x) for x ∈ [-2, 2]. Using our custom autograd engine, MSE loss, and the RMSprop optimizer, the network successfully learned to approximate the sine curve from scratch. The loss steadily decreased, and visualization showed the model’s predictions bending smoothly into the true sine wave — a clean sanity check that gradients, backpropagation, and optimization are working correctly.

<img width="568" height="413" alt="download" src="https://github.com/user-attachments/assets/b16bc470-da6d-4d7c-b8a6-bd045ca41ceb" />

## Visulization
For better visulization of neural network we used Graphviz library in `graph_visulization.ipynb` which produce minimal graph visulization. Below we tested it with minimal 2D nueron.
```
n = Neuron(2)
inputs =  [2.0, 1.0]
output = n(inputs)
output.backward()
draw_dot(output)
```
<img width="1869" height="303" alt="image" src="https://github.com/user-attachments/assets/1777eb4f-102e-433c-a99a-17f76c63a827" />
