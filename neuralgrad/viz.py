from graphviz import Digraph

def trace(root):
    """
    Traverse the computation graph starting from 'root'
    and collect nodes and edges.
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    """
    Visualize the computation graph using Graphviz.
    Each Value node shows: label | data | grad
    """
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        # Add node with label
        dot.node(
            name=uid,
            label="{%s | data: %.4f | grad: %.4f}" % (n.label, n.data, n.grad),
            shape="record",
        )

        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
