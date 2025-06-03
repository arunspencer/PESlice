# PESlice
A Python package for slicing high-dimensional potential energy surfaces, evaluating using [``graph-pes``](https://jla-gardner.github.io/graph-pes//) neural network models, and quantifying their complexity.

Developed during my Master's (Part II) project at the University of Oxford alongside my supervisor, John Gardner, in the Deringer Group.


## Features
* 🧪 **PES slicing** for structures using:
  * ``LinearSlicePES``: translate atoms linearly along chosen, or random, directions
  * ``CircularSlicePES``: move atoms along circular paths
* 🔢 **Complexity** quantification of evaluated slices via:
  * **Chebyshev** polynomial fitting using Lasso regression
  * **Mean Absolute Curvature (MAC)** from numerical gradients
* 📈 **Slice evaluation** via  neural networks from [``graph-pes``](https://jla-gardner.github.io/graph-pes//)


## Installation
Clone and install from GitHub:
```bash
pip install git+https://github.com/arunspencer/PESlice.git
```

This will install the package along with its dependencies, including: 
* ``numpy``, ``scipy``, ``torch``, ``ase``, ``scikit-learn``
* ``graph-pes`` – for loading NN models and evaluating slice trajectories
* ``jaxtyping`` – for smarter typing


## Quick Start
```python
from ase.build import molecule
from peslice import CircularSlicePES
from graph_pes.models import load_model

# Load structure
atoms = molecule("CH4")

# Define slice
slice = CircularSlicePES(atoms)
slice.def_slice(num_steps=200, max_radius=0.22, constant_radius=True)

# Load trained NN model
model = load_model("path/to/model.pt")

# Evaluate PES
slice.gp_eval(model, get_forces=True, batch_size=5)

# Extract energies and forces
energies = slice.energies  # shape (num_steps,)
forces = slice.forces  # shape (num_steps, num_atoms, 3)
abs_forces = slice.abs_forces  # shape (num_steps, num_atoms)

# Compute complexity
chebyshev_complexity = slice.cheb_complexity
mac_complexity = slice.mac_complexity
```


## Slicing Methods – ``peslice.core``

### ``LinearSlicePES``
Slice a potential energy surface (PES) along a n-dimensional hyperplane (n = 1, 2, 3) by moving each atom in a defined, or random, direction.    

The hyperplane is defined by: ``y = m0 + c1*m1 + c2*m2 + c3*m3 + ...``

Atomic positions are defined by: ``p(c1, c2, c3, ...) = p0 + c1*m1 + c2*m2 + c3*m3 + ...``

**Includes**:
* ``def_slice(...)`` – define a hyperplane
* ``def_rslice(...)`` – define a random hyperplane
* ``gp_eval(...)`` – evaluate a defined slice using a ``graph-pes`` neural network model
* ``cheb_complexity``, ``mac_complexity`` – calculate the complexity of an evaluated slice

### ``CircularSlicePES``
Slice a potential energy surface (PES) in 1D by moving each atom in circular paths.

**Includes**:
* ``def_slice(...)`` – define a 1D slice
* ``gp_eval(...)`` – evaluate a defined slice using a ``graph-pes`` neural network model
* ``cheb_complexity``, ``mac_complexity`` – calculate the complexity of an evaluated slice

## Complexity Measures – ``peslice.complexity``

### ``ChebyshevModel``
Fit Chebyshev polynomials to an n-dimensional function, ``y=f(X)``, using Lasso regression. Access coefficients, predict values using the fitted polynomial model, and calculate the Chebyshev complexity.

**Parameters**:
* ``alpha`` – the Lasso regularisation parameter
* ``order`` – the maximum Chebyshev polynomial order to expand

**Includes**:
* ``fit(X, y)`` – fit Chebyshev polynomials to a function ``y=f(X)``
* ``predict(X)`` – predict y from X using the Chebyshev polynomial fit
* ``complexity`` – calculate the Chebyshev complexity
* ``coefficients`` – returns the fitted Chebyshev polynomial coefficients

### ``chebyshev_complexity.chebyshev_complexity``
Returns the Chebyshev complexity of an n-dimensional function, ``y=f(X)``.

**Parameters**:
* ``X``, ``y`` – the function ``y=f(X)``
* ``alpha`` – the Lasso regularisation parameter
* ``max_order`` – the maximum Chebyshev polynomial order to expand

### ``mac_complexity.mac_complexity``
Returns the MAC complexity of a 1D function, ``y=f(x)``.

**Parameters**
* ``y`` – y values in the function ``y=f(x)``
* ``dx`` – discrete spacing between points in ``y=f(x)``
* ``normalise`` – Whether to normalise the MAC complexity by the standard deviation of y


## Utilities – ``peslice.utils``

### ``geometry.check_orthogonal``
Checks the orthogonality of Cartesian sub-vectors of parsed vectors for use in ``LinearSlicePES``.

### ``geometry.gen_random_slice``
Returns a random 1-, 2-, or 3-dimensional orthonormal basis to define a hyperplane for use in ``LinearSlicePES``.

Orthonormality is checked per sub-vector.

### ``geometry.normalised``
Normalise each row vector of a parsed tensor.

### ``graphs.define_graphs``
Creates trimmed atomic graphs for each structure using a shared large neighbour list.

Avoids recomputing full neighbour lists for similar structures.

### ``graphs.get_nearest_neighbour_distance``
Returns the smallest nearest neighbour distance in an atomic graph.

### ``graphs.batch_calculate``
Batch calculates the energies and (optionally) forces for each atomic graph.