# Micrograd

Micrograd is a notebook-first tour of automatic differentiation. The project
starts with derivatives calculated numerically, then builds the pieces needed
to represent a mathematical expression as a graph and propagate gradients
back through it.

The main tutorial is [`main.ipynb`](main.ipynb).

## What You Will Learn

- How a derivative measures the effect of a small change in an input.
- How finite differences approximate derivatives and partial derivatives.
- How a `Value` object can store a scalar and the nodes that produced it.
- How an expression graph makes the chain rule easier to apply.
- How gradients can be used to nudge values toward a larger output.

## Run The Notebook

The repository includes `uv` metadata and targets Python 3.14 or newer. The
current checkout is notebook-first and does not yet contain the
`src/micrograd` package declared by `pyproject.toml`, so use `uv` without
building the project itself:

```bash
uv run --no-project \
  --with jupyterlab \
  --with matplotlib \
  --with numpy \
  --with networkx \
  jupyter lab main.ipynb
```

Alternatively, use a regular virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab matplotlib numpy networkx
jupyter lab main.ipynb
```

Run the cells from top to bottom. Several cells reuse names such as `a`, `b`,
and `d`, and the notebook uses `%reset -f` between stages.

## Tutorial

### 1. Start With A Derivative

The notebook first plots the function:

```python
def f(x):
    return 3 * x**2 - 4 * x + 5
```

At `x = 3`, a small step `h` gives the finite-difference approximation:

```python
h = 0.001
x = 3.0
slope = (f(x + h) - f(x)) / h
```

The result is approximately `14`, the slope of the curve at that point. The
notebook also checks that the slope is approximately zero at `x = 2/3`.

### 2. Move To Multiple Inputs

For a function with several inputs, change one input at a time while keeping
the others fixed:

```python
a = 2.0
b = -3.0
c = 10.0
d = a * b + c
```

Numerically nudging `a`, `b`, and `c` gives the partial derivatives:

```text
dd/da = -3
dd/db =  2
dd/dc =  1
```

This is the basic idea behind a gradient: a list of partial derivatives that
tells us how each input affects the output.

### 3. Wrap Numbers In `Value` Objects

The next step is to keep more than a number. A `Value` node stores its scalar
data and, when it is created by an operation, remembers its parents:

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")
```

Now an expression such as `d = a * b + c` retains both its numerical result
and the structure that produced it. The notebook adds labels and uses
`networkx` and Matplotlib to draw this structure.

### 4. Apply The Chain Rule By Hand

The notebook uses this small graph:

```text
e = a * b
d = e + c
L = d * f
```

With `a = 2`, `b = -3`, `c = 10`, and `f = -2`, the forward pass gives
`L = -8`.

Starting at the output, set `dL/dL = 1` and work backward. The local
derivatives are:

```text
dL/dd = f = -2
dL/df = d =  4
dL/dc = dL/dd * 1 = -2
dL/de = dL/dd * 1 = -2
dL/da = dL/de * b =  6
dL/db = dL/de * a = -4
```

The important rule is:

```text
gradient at a node = gradient from above * local derivative
```

That repeated multiplication is the chain rule in the form used by reverse-
mode automatic differentiation.

### 5. Use A Gradient To Change The Output

Once every leaf has a gradient, update it by a small step. For gradient ascent
with learning rate `eta`:

```python
eta = 0.01
a.data += eta * a.grad
b.data += eta * b.grad
c.data += eta * c.grad
f.data += eta * f.grad
```

Moving each value in the direction of its gradient makes `L` larger for a
sufficiently small step. Neural-network training uses the same idea, usually
with gradient descent to make a loss smaller.

## Notebook Note

The first `SNEAK PEAK` cell imports `Value` with:

```python
from micrograd.engine import Value
```

This checkout is currently an educational notebook rather than a separately
packaged `micrograd` module. If that import is unavailable in your environment,
skip the preview cell and start with **DERIVATIVE SIMPLER FUNCTION**; the
notebook defines the teaching version of `Value` later on.

## Project Layout

```text
.
├── main.ipynb       # The complete autodiff tutorial
├── pyproject.toml   # Project metadata and dependencies
├── uv.lock          # Locked uv dependencies
└── LICENSE          # GPL-3.0 license
```

## License

This project is distributed under the GNU General Public License v3. See
[`LICENSE`](LICENSE) for the full text.
