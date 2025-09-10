"""
NeuralGrad: Minimal autograd + neural networks.
"""

__version__ = "0.1.0"

# Core
from .engine import Value

# Visualization
from .viz import draw_dot, trace

# Neural network layers
from .nn import Neuron, Layer, MLP

# Loss functions
from .losses import MSE, CrossEntropy

# Optimizers
from .optim import RMSprop, Adam

__all__ = [
    "Value",
    "draw_dot",
    "trace",
    "Neuron",
    "Layer",
    "MLP",
    "MSE",
    "CrossEntropy",
    "RMSprop",
    "Adam",
]
