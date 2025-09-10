from neuralgrad import Value, draw_dot

# inputs
x1 = Value(2.0, label="x1")
x2 = Value(-3.0, label="x2")

# weights
w1 = Value(0.0, label="w1")
w2 = Value(1.0, label="w2")

# bias
b = Value(4.181, label="b")

# forward pass
n = x1 * w1 + x2 * w2 + b
o = n.sigmoid()
o.label = "output"

# backward
o.backward()

# visualize
dot = draw_dot(o)
dot.render("basic_autograd", view=True)
