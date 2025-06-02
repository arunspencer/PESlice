# ruff: noqa: F821, F722


import numpy as np
import ase
import torch
from graph_pes import GraphPESModel
from peslice.utils.geometry import normalised
from peslice.utils.graphs import define_graphs, batch_calculate


class CircularSlicePES:
    """
    Used to slice a potential energy surface (PES) in 1D by moving each atom in circular paths with bounding radii and angular speeds

    The 1D slice is defined by moving through time

    PES slices can be evaluted using a parsed `graph-pes` network model

    Built-in PES complexity calculations using Chebyshev and mean absolute curvature (MAC) methods

    Parameters
    ----------
    structure
        The `ase.Atoms` object of the structure being studied
    cheb_alpha
        The alpha value for the Lasso regression in the Chebyshev polynomial fitting model

        Default value = 0.0005

    Attributes
    ----------
    structure
        The `ase.Atoms` object of the structure being studied
    slice_structures
        The `ase.Atoms` objects for each sample point along the 1D slice
    t
        The time points for each sample point along the 1D slice
    cheb_score
        The R^2 score of the Chebyshev fit
    cheb_complexity
        The Chebyshev complexity measure of the PES slice - see `chebyshev.py`
    mac_complexity
        The mean absolute curvature complexity measure of the PES slice - see `mac_complexity.py`
    energies
        The energies of the sample points in the hyperplane calculated by the parsed `graph-pes` model
    forces
        The atomic forces of the sample points in the hyperplane calculated by the parsed `graph-pes` model
    abs_forces
        The absolute atomic forces of the sample points in the hyperplane calculated by the parsed `graph-pes` model
    steps
    The steps along the 1D slice (a list from 1 to the number of steps)

    Methods
    -------
    def_slice
        Defines the 1D slice by moving each atom in circular paths with bounding radii and angular speeds
    gp_eval
        Evaluates the energy of the `ase.Atoms` objects at each sample point along the 1D slice using a parsed `graph-pes` network model
    """

    def __init__(
        self,
        structure: ase.Atoms,
        cheb_alpha: float = 0.0005,
        overwrite_warning_supressed: bool = False,
        persistent_info: bool = True,
    ):
        self._defined = False
        self._evaluated = False
        self._cheb_alpha = cheb_alpha
        self.structure = structure
        self.overwrite_warning_supressed = overwrite_warning_supressed
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
        Defines a 1D slice by moving each atom in circular paths with bounding radii and angular speeds

        Parameters
        ----------
        model_cutoff
            The cutoff distance for the graph-pes model

            Default value = 5.0
        min_radius
            The minimum radius of the circular path

            Default value = 0.01
        max_radius
            The maximum radius of the circular path

            Default value = 0.08
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
            The seed for the random generation of the slice

            Default value = 42
        """
        # checks if a slice is already defined - overwrites
        if self._defined:
            if not self.overwrite_warning_supressed:
                print("Overwriting previous slice")
            self._evaluated = False
            self.t = np.zeros_like(self.t)

        # checks if the slice is already evaluated - removes previous evaluations
        if self._evaluated:
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

    def gp_eval(
        self,
        model: GraphPESModel,
        get_forces: bool = False,
        batch_size: int = 10,
    ) -> None:
        """
        Evaluates the energy, and optionally forces, of the ase.Atoms objects at each sample point along the 1D slice

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
        """
        if not self._defined:
            raise ValueError("Slice must be defined before it can be evaluated")

        energies, forces, abs_forces = batch_calculate(
            model, self.graphs, get_forces=get_forces, batch_size=batch_size
        )

        self.energies = energies
        self.forces = forces
        self.abs_forces = abs_forces
        self._evaluated = True

    def _cheb_input(self) -> np.ndarray:
        return np.array(self.t)
    
    def _mac_dx(self) -> float:
        return float(self.t[-1]) / len(self.t)
    
    def __repr__(self) -> str:
        return (
            "TODO"
        )

    def summary(self) -> str:
        return (
            "TODO"
        )