# ruff: noqa: F821, F722


import numpy  as np
import ase
from jaxtyping import Float
from graph_pes import GraphPESModel
from peslice.core.base_slice import BaseSlicePES
from peslice.utils.geometry import gen_random_slice, check_orthogonal
from peslice.utils.graphs import define_graphs, batch_calculate, get_nearest_neighbour_distance

class LinearSlicePES(BaseSlicePES):
    """
    Used to slice a potential energy surface (PES) along a n-dimensional hyperplane (n=1, 2, 3)

    Hyperplanes are defined by a point and n-1 basis vectors (the vectors contain orthogonal subvectors of length 3 - corresponding to atomic (x, y, z) coordinates)

    PES slices can be evaluted using a parsed `graph-pes` network model

    The hyperplane is defined by the equation: y = m0 + c1*m1 + c2*m2 + c3*m3 + ..., such that atomic positions can be generated through the below equation

    Atomic positions: p(c1, c2, c3, ...) = p0 + c1*m1 + c2*m2 + c3*m3 + ...

    The coefficients have units of Angstroms (Å)

    Built-in PES complexity calculations using Chebyshev and mean absolute curvature (MAC) methods


    Parameters
    ----------
    structure
        The `ase.Atoms` object of the structure being studied
    point_density
        The number of points to sample along each dimension in the hyperplane per unit length

        Default value = 100
    cheb_alpha
        The alpha value for the Lasso regression in the Chebyshev polynomial fitting model

        Default value = 0.0005

    Attributes
    ----------
    function_dim
        The dimension of the function describing the PES
    point_density
        The number of points to sample along each dimension in the hyperplane per unit length
    ranges_n
        The value of n to define [-n, n] ranges for the plane coefficients
    slice_dim
        The dimension of the hyperplane
    m0
        A point on the hyperplane

        Shape `(function_dim, )`
    orthogonal_basis
        The orthogonal basis vectors that define the hyperplane `(m1, m2, m3, ...)`

        `y = m0 + c1*m1 + c2*m2 + c3*m3 + ...`

        Shape `(slice_dim, function_dim)`
    coefficients
        The values of `(c1, c2, c3, ...)` that define the sample points in the hyperplane

        Shape `(num_points, slice_dim)`

        `num_points` is the product of the number of points in each dimension, determined by the `point_density` parameter
    structure
        The `ase.Atoms` object of the structure being studied
    slice_structures
        The `ase.Atoms` objects for each sample point along the hyperplane
    graphs
        The `graph_pes.AtomicGraph` objects for each sample point along the hyperplane
    complexity
        The Chebyshev complexity measure of the PES slice - see `chebyshev.py`
    cheb_score
        The R^2 score of the Chebyshev fit
    energies
        The energies of the sample points in the hyperplane calculated by the parsed `graph-pes` model
    forces
        The atomic forces of the sample points in the hyperplane calculated by the parsed `graph-pes` model
    abs_forces
        The absolute atomic forces of the sample points in the hyperplane calculated by the parsed `graph-pes` model
    nn_neighbour_distances
        The nearest nearest neighbour distance for each sample point along the hyperplane
    Methods
    -------
    def_slice
        Defines the hyperplane in `slice_dim`-dimensions by parsing orthogonal basis vectors
    def_rslice
        Defines a random hyperplane in `slice_dim`-dimensions
    gp_eval
        Evaluates the function at the sample points in the hyperplane using a parsed `graph-pes` many-body model
    """

    def __init__(
        self,
        structure: ase.Atoms,
        point_density: int = 100,
        cheb_alpha: float = 0.0005,
        get_nn_distances: bool = False,
    ):
        super().__init__(structure=structure, cheb_alpha=cheb_alpha)
        self.point_density = point_density
        self.get_nn_distances = get_nn_distances
        self._get_forces = False

    def def_slice(
        self,
        m0: Float[np.ndarray, "function_dim "],
        *args: Float[np.ndarray, "function_dim "],
        model_cutoff: float = 5.0,
        ranges_n: float = 0.9,
    ) -> None:
        """
        Defines the hyperplane in n-dimensions (n=1, 2, 3)

        Parameters
        ----------
        m0
            A point on the hyperplane

            Must have `dtype=np.float64`

            Shape `(function_dim, )`
        *args
            Orthogonal basis vectors that define the hyperplane `(m1, m2, m3, ..., mA)`

            Must have `dtype=np.float64`

            Shape `(function dim, )`
        model_cutoff
            The cutoff distance for the graph-pes model

            Default value = 5.0
        ranges_n
            The value of n to define [-n, n] ranges for the plane coefficients

            Default value = 0.9

        Raises
        ------
        ValueError
            If the basis vectors are not orthogonal in each sub-vector

            If ranges_n is <= 0

            If the basis vectors are not non-zero

            If the length of the vector is not (3*number of atoms) - atomic (x, y, z) coordinates

            If the vectors do not have `dtype=np.float64`

            If m0 does not have `dtype=np.float64`
        """
        # checks that ranges_n is > 0
        if ranges_n <= 0:
            raise ValueError("ranges_n must be > 0")

        # checks that the vectors are non-zero
        for vec in args:
            norms = np.linalg.norm(vec.reshape(-1, 3), axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("The vectors must be non-zero")

        # checks that for each vec in args, that the dtype is float64
        for vec in args:
            if vec.dtype != np.float64:
                raise ValueError("The vectors must have `dtype=np.float64`")

        # checks that m0 is of dtype float64
        if m0.dtype != np.float64:
            raise ValueError("m0 must have `dtype=np.float64`")

        # checks if the vectors are orthogonal if not already checked
        if len(args) > 1:
            if not check_orthogonal(*args, function_dim=len(m0)):
                raise ValueError("The vectors must be orthogonal")

        # checks that the number of len(m0) == 3*number of atoms
        if len(m0) != 3 * len(self.structure):
            raise ValueError(
                "The length of the vector must equal (3*number of atoms) - atomic (x, y, z) coordinates"
            )

        # creates the orthogonal basis vectors
        slice_dim = len(args)
        function_dim = len(m0)

        orthogonal_basis = np.zeros((slice_dim, function_dim))
        for i, vec in enumerate(args):
            vec = vec.reshape(-1, 3)
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
            orthogonal_basis[i, :] = vec.reshape(-1)

        if self._defined:
            self._evaluated = False
            self.energies = np.zeros_like(self.energies)
            if self._get_forces:
                self.forces = np.zeros_like(self.forces)

        self.orthogonal_basis = orthogonal_basis
        self.slice_dim = slice_dim
        self.function_dim = function_dim
        self.m0 = m0
        self.ranges_n = ranges_n

        # generates sample points in the hyperplane
        self._generate_points()

        # defines ase.Atoms objects for each sample point
        self.slice_structures = self._generate_atoms()

        # defines AtomicGraph objects for each sample point
        self.graphs = define_graphs(
            self.structure, self.slice_structures, model_cutoff
        )

        self._defined = True

        # get nearest neighbour distances
        nn_neighbour_distances = []
        for graph in self.graphs:
            nn_neighbour = get_nearest_neighbour_distance(graph)
            nn_neighbour_distances.append(nn_neighbour)

        self.nn_neighbour_distances = nn_neighbour_distances

    def def_rslice(
        self,
        slice_dim: int,
        model_cutoff: float = 5.0,
        seed: int = 42,
        ranges_n: float = 0.9,
    ) -> None:
        """
        Defines a random hyperplane in `slice-dim`-dimensions.

        Parameters
        ----------
        slice_dim
            The dimension of the hyperplane
        model_cutoff
            The cutoff distance for the graph-pes model

            Default value = 5.0
        seed
            The seed for the random generation of slices
        ranges_n
            The value of n to define [-n, n] ranges for the plane coefficients

            Default value = 0.9

        Raises
        ------
        ValueError
            If the slice dimension is greater than the function dimension or is < 1

            If ranges_n is <= 0

            If m0 does not have `dtype=np.float64`
        """

        m0 = self.structure.positions.reshape(-1)

        function_dim = len(m0)

        # checks that ranges_n is > 0
        if ranges_n <= 0:
            raise ValueError("ranges_n must be > 0")

        # checks that the slice dimension is in the range [1, function_dim]
        if slice_dim < 1 or slice_dim > 3:
            raise ValueError("Slice dimension must be in the range [1, 3]")

        # checks that the length of the vector is (3*number of atoms) - atomic (x, y, z) coordinates
        if len(m0) != 3 * len(self.structure):
            raise ValueError(
                "The length of the vector must equal (3*number of atoms) - atomic (x, y, z) coordinates"
            )

        # checks that the dtype of m0 is float64
        if m0.dtype != np.float64:
            raise ValueError("m0 must have `dtype=np.float64`")

        # overwrites the previous slice if it is already defined
        if self._defined:
            print("Overwriting previous slice")
            self._evaluated = False
            self.energies = np.zeros_like(self.energies)
            if self._get_forces:
                self.forces = np.zeros_like(self.forces)

        # creates orthogonal basis vectors
        orthogonal_basis = gen_random_slice(m0, slice_dim, seed)

        self.orthogonal_basis = orthogonal_basis
        self.slice_dim = slice_dim
        self.function_dim = function_dim
        self.m0 = m0
        self.ranges_n = ranges_n

        # generates sample points in the hyperplane
        self._generate_points()

        # defines ase.Atoms objects for each sample point
        self.slice_structures = self._generate_atoms()

        # defines AtomicGraph objects for each sample point
        self.graphs = define_graphs(
            self.structure, self.slice_structures, model_cutoff
        )

        self._defined = True

        # get nearest neighbour distances
        nn_neighbour_distances = []
        for graph in self.graphs:
            nn_neighbour = get_nearest_neighbour_distance(graph)
            nn_neighbour_distances.append(nn_neighbour)

        self.nn_neighbour_distances = nn_neighbour_distances

    def _generate_points(self) -> None:
        """
        Generates sample points in the hyperplane from the orthogonal basis vectors, `ranges_n`, and `point_density` (in each dimension)
        """
        slice_dim = self.slice_dim
        ranges_n = self.ranges_n

        # generating sample points in each hyperplane dimension (in terms of the coefficients)
        sample_points = self.point_density * (2 * ranges_n)

        ranges_points = [
            np.linspace(-ranges_n, ranges_n, int(sample_points))
            for _ in range(slice_dim)
        ]

        coefficients = np.vstack(
            [x.ravel() for x in np.meshgrid(*ranges_points)]
        ).T  # stores coordinates in c1, c2, c3, ... on the rows

        self.coefficients = coefficients

    def _generate_atoms(self) -> list[ase.Atoms]:
        molecules = []  # to store an Atoms object for each sample point

        # generating new atomic positions for each sample point, creating a new Atoms object
        structure = self.structure
        symbols = structure.get_chemical_symbols()
        for j, coeff_points in enumerate(self.coefficients):
            new_atomic_positions = structure.positions.reshape(-1) + np.sum(
                [
                    coeff_points[i] * self.orthogonal_basis[i]
                    for i in range(self.slice_dim)
                ],
                axis=0,
            )

            molecules.append(
                ase.Atoms(
                    symbols=symbols,
                    positions=new_atomic_positions.reshape(
                        structure.positions.shape
                    ),
                    pbc=structure.pbc,
                    cell=structure.cell,
                )
            )

        return molecules

    def gp_eval(
        self,
        model: GraphPESModel,
        get_forces: bool = False,
        batch_size: int = 10,
    ) -> None:
        """
        Evaluates the energy, and optionally forces, of the ase.Atoms objects at each sample point along the hyperplane

        Parameters
        ----------
        model
            The `graph-pes` network model class instance to use
        get_forces
            Whether to calculate the forces at each sample point

            Default value = False
        batch_size
            The batch size to use for the evaluation of energies and (optionally) forces

            Default value = 10

        Raises
        ------
        ValueError
            If the slice has not yet been defined

            If the number of atomic directions in the structure does not match the function dimension
        """
        if not self._defined:
            raise ValueError("Slice must be defined before it can be evaluated")

        if self._evaluated:
            self._evaluated = False

        self._get_forces = get_forces

        energies, forces, abs_forces = batch_calculate(
            model, self.graphs, batch_size=batch_size, get_forces=get_forces
        )

        self.energies = energies  # shape (num_points, )
        self.forces = forces
        self.abs_forces = abs_forces

        self._evaluated = True

    def _cheb_input(self) -> np.ndarray:
        return self.coefficients
    
    def _mac_dx(self) -> float:
        return 1/self.point_density
    
    def __repr__(self) -> str:
        return (
            "TODO"
        )

    def summary(self) -> str:
        return (
            "TODO"
        )