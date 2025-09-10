import torch
import math
import pytest
from neuralgrad import Value

def test_addition_and_grad():
    # NeuralGrad
    x = Value(3.0)
    y = Value(-2.0)
    z = x + y + 5
    z.backward()
    grad_ng = (x.grad, y.grad, z.data)

    # PyTorch
    xt = torch.tensor([3.0], dtype=torch.double, requires_grad=True)
    yt = torch.tensor([-2.0], dtype=torch.double, requires_grad=True)
    zt = xt + yt + 5
    zt.backward()
    grad_pt = (xt.grad.item(), yt.grad.item(), zt.item())

    assert math.isclose(grad_ng[2], grad_pt[2], rel_tol=1e-6)
    assert math.isclose(grad_ng[0], grad_pt[0], rel_tol=1e-6)
    assert math.isclose(grad_ng[1], grad_pt[1], rel_tol=1e-6)


def test_sigmoid_vs_torch():
    # NeuralGrad
    x = Value(0.5)
    y = x.sigmoid()
    y.backward()
    grad_ng = (y.data, x.grad)

    # PyTorch
    xt = torch.tensor([0.5], dtype=torch.double, requires_grad=True)
    yt = torch.sigmoid(xt)
    yt.backward()
    grad_pt = (yt.item(), xt.grad.item())

    assert math.isclose(grad_ng[0], grad_pt[0], rel_tol=1e-6)
    assert math.isclose(grad_ng[1], grad_pt[1], rel_tol=1e-6)


def test_chain_of_ops():
    # NeuralGrad: (x * y + z).relu()
    x = Value(-3.0)
    y = Value(2.0)
    z = Value(1.0)
    q = (x * y + z).relu()
    q.backward()
    grad_ng = (q.data, x.grad, y.grad, z.grad)

    # PyTorch
    xt = torch.tensor([-3.0], dtype=torch.double, requires_grad=True)
    yt = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
    zt = torch.tensor([1.0], dtype=torch.double, requires_grad=True)
    qt = torch.relu(xt * yt + zt)
    qt.backward()
    grad_pt = (qt.item(), xt.grad.item(), yt.grad.item(), zt.grad.item())

    assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(grad_ng, grad_pt))
