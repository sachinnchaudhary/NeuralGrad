class RMSprop:
    def __init__(self, parameters, lr=0.001, decay_rate=0.9, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.decay_rate = decay_rate
        self.eps = eps
        self.v = [0.0] * len(parameters)

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.v[i] = self.decay_rate * self.v[i] + (1 - self.decay_rate) * (param.grad ** 2)
                adaptive_grad = param.grad / (self.v[i] ** 0.5 + self.eps)
                param.data -= self.lr * adaptive_grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.0


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            g = param.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / ((v_hat ** 0.5) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.0
