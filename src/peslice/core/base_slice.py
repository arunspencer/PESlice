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
    ...
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
        if not self._defined:
            raise ValueError("Slice must be defined before complexity can be calculated")
        if not self._evaluated:
            raise ValueError("Slice must be evaluated before complexity can be calculated")

        model = ChebyshevModel(alpha=self._cheb_alpha)
        model.fit(self._cheb_input(), self.energies - self.energies.mean())
        self.cheb_score = model.score(self._cheb_input(), self.energies)
        return model.complexity

    @property
    def mac_complexity(self) -> float:
        if not self._defined:
            raise ValueError("Slice must be defined before complexity can be calculated")
        if not self._evaluated:
            raise ValueError("Slice must be evaluated before complexity can be calculated")

        return mac_complexity(self.energies, dx=self._mac_dx(), normalise=True)