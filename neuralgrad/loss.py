from .engine import Value

class MSE:
    def __init__(self, inputs, targets):
        self.predictions = inputs
        self.targets = targets

    def compute_loss(self):
        squared_errors = [(pred - target) ** 2 for pred, target in zip(self.predictions, self.targets)]
        total_loss = sum(squared_errors)
        return total_loss / len(self.predictions)


class CrossEntropy:
    def __init__(self, inputs, targets):
        self.predictions = inputs
        self.targets = targets

    def softmax(self):
        exp = [i.exp() for i in self.predictions]
        summed_exp = sum(exp)
        return [e / summed_exp for e in exp]

    def compute_loss(self):
        if isinstance(self.targets, int):
            one_hot = [0] * len(self.predictions)
            one_hot[self.targets] = 1
            targets = one_hot
        else:
            targets = self.targets

        probs = self.softmax()
        loss = Value(0.0)
        for pred, target in zip(probs, targets):
            if target == 1:
                epsilon = 1e-15
                safe_pred = pred + epsilon
                loss = loss + (-1 * target * safe_pred.log())
        return loss
