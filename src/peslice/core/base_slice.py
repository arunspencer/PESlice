# ruff: noqa: F821, F722


from abc import ABC, abstractmethod
import ase
import numpy as np
from graph_pes import GraphPESModel
from peslice.utils.graphs import batch_calculate
from peslice.complexity.chebyshev_complexity import ChebyshevModel
from peslice.complexity.mac_complexity import mac_complexity


class BaseSlicePES(ABC):
    """
    Base class for PES slicing methods

    Provides the basic structure for defining PES slices, evaluating using ``graph-pes`` models, and calculating complexities

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

    Methods
    -------
    def_slice
        Defines the slice by implementing the specific slicing method in a subclass
    gp_eval
        Evaluates the energy of the ``ase.Atoms`` objects at each sample point along the slice using a parsed ``graph-pes`` NN model
    cheb_complexity
        Calculates the Chebyshev complexity of the slice, requires the slice to be defined and evaluated prior to calling
    mac_complexity
        Calculates the mean absolute curvature complexity of the slice, requires the slice to be defined and evaluated prior to calling
    """
    def __init__(self, structure: ase.Atoms, cheb_alpha: float = 0.0005):
        self.structure = structure
        self._cheb_alpha = cheb_alpha
        self._defined = False
        self._evaluated = False
        self.graphs = []


    @abstractmethod
    def def_slice(self, *args, **kwargs) -> None:
        pass

    def gp_eval(
        self,
        model: GraphPESModel,
        get_forces: bool = False,
        batch_size: int = 10,
    ) -> None:
        """
        Evaluates the energy and (optionally) forces of the ``ase.Atoms`` objects at each sample point along the slice trajectory

        Parameters
        ----------
        model
            The ``graph-pes`` NN model to use
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
            raise ValueError("Slice must be defined before evaluation")

        energies, forces, abs_forces = batch_calculate(
            model, self.graphs, batch_size=batch_size, get_forces=get_forces
        )

        self.energies = energies
        self.forces = forces if get_forces else None
        self.abs_forces = abs_forces if get_forces else None
        self._evaluated = True

    @abstractmethod
    def _cheb_input(self) -> np.ndarray:
        pass

    @abstractmethod
    def _mac_dx(self) -> float:
        pass

    @property
    def cheb_complexity(self) -> float:
        if not self._defined or not self._evaluated:
            raise ValueError("Slice must be defined and evaluated before complexity can be calculated")

        model = ChebyshevModel(alpha=self._cheb_alpha)
        model.fit(self._cheb_input(), self.energies - self.energies.mean())
        self.cheb_score = model.score(self._cheb_input(), self.energies)
        return model.complexity

    @property
    def mac_complexity(self) -> float:
        if not self._defined or not self._evaluated:
            raise ValueError("Slice must be defined and evaluated before complexity can be calculated")

        return mac_complexity(self.energies, dx=self._mac_dx(), normalise=True)