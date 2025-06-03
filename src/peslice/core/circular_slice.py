# ruff: noqa: F821, F722


import numpy as np
import ase
import torch
from peslice.core.base_slice import BaseSlicePES
from peslice.utils.geometry import normalised
from peslice.utils.graphs import define_graphs


class CircularSlicePES(BaseSlicePES):
    """
    Slice a potential energy surface (PES) in 1D by moving each atom in circular paths with a random (or constant) radius but random angular speeds and rotation planes

    All random parameters are bounded by the parameters defined in the ``def_slice`` method

    The 1D slice is defined by moving through an arbitrary time dimension

    PES slices can be evaluated using a parsed ``graph-pes`` NN model

    Built-in PES complexity calculations using Chebyshev and mean absolute curvature (MAC) methods

    Parameters
    ----------
    structure
        The ``ase.Atoms`` object of the structure being studied
    cheb_alpha
        The alpha value for the Lasso regression in the Chebyshev polynomial fitting model

        Default value = 0.0005

    Attributes
    ----------
    structure
        The ``ase.Atoms`` object of the structure being studied
    graphs
        The list of graphs representing the slice points, defined by the ``def_slice`` method
    energies
        The energies of the sample points in the hyperplane calculated by the parsed ``graph-pes`` model, defined in the ``gp_eval`` method
    forces
        The atomic forces of the sample points in the hyperplane calculated by the parsed ``graph-pes`` model, defined in the ``gp_eval`` method

        Default value = None
    abs_forces
        The absolute atomic forces of the sample points in the hyperplane calculated by the parsed ``graph-pes`` model, defined in the ``gp_eval`` method

        Default value = None
    cheb_score
        The R^2 score of the Chebyshev fit, calculated in the ``cheb_complexity`` property
    cheb_complexity
        The Chebyshev complexity measure of the PES slice - ``peslice.complexity.chebyshev_model``
    mac_complexity
        The mean absolute curvature complexity measure of the PES slice - ``peslice.complexity.mac_complexity``
    slice_structures
        The ``ase.Atoms`` objects for each sample point along the hyperplane
    persistent_info
        Whether to keep ``ase.Atoms.info`` attributes for each sample point along the hyperplane
    t
        The arbitrary time dimension of the 1D slice, defined by the ``def_slice`` method
    steps
        The list of steps along the 1D slice, defined by the ``def_slice`` method

    Methods
    -------
    def_slice
        Defines a 1D slice by moving each atom in circular paths

    gp_eval
        Evaluates the energy and (optionally) forces of the ``ase.Atoms`` objects at each sample point along the 1D slice using a parsed ``graph-pes`` NN model
    """

    def __init__(
        self,
        structure: ase.Atoms,
        cheb_alpha: float = 0.0005,
        persistent_info: bool = True,
    ):
        super().__init__(structure=structure, cheb_alpha=cheb_alpha)
        self.persistent_info = persistent_info
        self._get_forces = False

    def def_slice(
        self,
        model_cutoff: float = 5.0,
        min_radius: float = 0.001,
        max_radius: float = 0.2,
        constant_radius: bool = True,
        max_angular_speed: float = 1.0,
        n_steps: int = 250,
        max_t: float = 5.0,
        seed: int = 42,
    ) -> None:
        """
        Defines a 1D slice by moving each atom in circular paths with a constant radius but random angular speeds and rotation planes

        Parameters
        ----------
        model_cutoff
            The cutoff distance for the graph-pes model

            Default value = 5.0
        min_radius
            The minimum radius of the circular path

            Default value = 0.001
        max_radius
            The maximum radius of the circular path

            Default value = 0.2
        constant_radius
            Whether to use the maximum radius as a constant radius for all atoms

            Default value = True
        max_angular_speed
            The maximum angular speed of the circular path

            Default value = 1.0
        n_steps
            The number of steps to take along the circular path

            Default value = 100
        max_t
            The maximum time to take along the circular path

            Default value = 5.0
        seed
            The seed for random number generation

            Default value = 42
        """
        # checks if a slice is already defined
        if self._defined:
            self._evaluated = False

        R = torch.tensor(self.structure.positions)

        gen = torch.Generator()
        gen.manual_seed(seed)

        def rand(*args):
            return torch.rand(*args, generator=gen)

        dt = max_t / n_steps

        t = torch.arange(n_steps) * dt

        # get 2 orthonormal directions
        dir_x = normalised(rand(R.shape))
        dir_y = normalised(rand(R.shape))
        dir_y -= dir_x * (dir_y * dir_x).sum(dim=1, keepdim=True)
        dir_y = normalised(dir_y)

        # get random radii and speeds
        if constant_radius:
            radii = torch.full((R.shape[0],), max_radius)
        else:
            radii = (
                rand(R.shape[0]) * (max_radius - min_radius) + min_radius
            )  # (N,)

        speeds = rand(R.shape[0]) * max_angular_speed * torch.pi  # (N,)

        # get thetas
        thetas = speeds[..., None] * t  # (N, T)
        thetas = thetas.transpose(0, 1)  # (T, N)

        # define t and positions
        r, path = (
            t,
            R[None, ...]  # (1, N, d)
            + radii[None, ..., None]
            * (  # (1, N, 1)
                torch.cos(thetas)[..., None] * dir_x[None, ...]  # (N, T, d)
                + torch.sin(thetas)[..., None] * dir_y[None, ...]  # (N, T, d)
            ),
        )

        # generating ase.Atoms objects for each sample point
        slice_structures = []
        symbols = self.structure.get_chemical_symbols()

        if self.persistent_info:
            for pos in path:
                slice_structures.append(
                    ase.Atoms(
                        symbols=symbols,
                        positions=pos,
                        pbc=self.structure.pbc,
                        cell=self.structure.cell,
                        info=self.structure.info,
                    )
                )
        else:
            for pos in path:
                slice_structures.append(
                    ase.Atoms(
                        symbols=symbols,
                        positions=pos,
                        pbc=self.structure.pbc,
                        cell=self.structure.cell,
                    )
                )

        self.slice_structures = slice_structures

        # ase.Atoms -> graph_pes.AtomicGraph objects
        self.graphs = define_graphs(
            self.structure, self.slice_structures, model_cutoff
        )

        self.t = r
        self.steps = [i + 1 for i in range(n_steps)]
        self._defined = True

    def _cheb_input(self) -> np.ndarray:
        return np.array(self.t)
    
    def _mac_dx(self) -> float:
        return float(self.t[-1]) / len(self.t)