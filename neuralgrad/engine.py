import math

class Value:
    """
    Scalar value that supports automatic differentiation.
    """

    def __init__(self, data, _child=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_child)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # Basic arithmetic
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    # Activations
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        r = x if x > 0 else 0
        out = Value(r, (self,), "relu")

        def _backward():
            self.grad += out.grad * (1 if x > 0 else 0)
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += out.grad * s * (1 - s)
        out._backward = _backward
        return out

    # Other ops
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        x = self.data
        assert x > 0
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward
        return out

    # Backprop
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()
