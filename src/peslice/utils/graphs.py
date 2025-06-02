import numpy as np
import ase
import torch
from graph_pes import GraphPESModel
from graph_pes.atomic_graph import AtomicGraph, trim_edges
from graph_pes.data import GraphDataLoader
from warnings import warn


def define_graphs(
    starting_molecule: ase.Atoms,
    molecules: list[ase.Atoms],
    model_cutoff: float,
    large_cutoff: float = 9.0,
) -> list[AtomicGraph]:
    """
    Creates trimmed atomic graphs for each structure using a shared large neighbour list
    
    Neighbour list calculated from the ``starting_molecule``, which is _similar_ to the structures in ``molecules``

    Each structure in ``molecules`` is trimmed to the ``model_cutoff`` distance

    This avoids recomputing full neighbour lists for each structure

    Parameters
    ----------
    starting_molecule
        The reference molecule to define the large neighbour list
    molecules
        List of molecules to create atomic graphs for
    model_cutoff
        The cutoff radius for trimming edges in the atomic graphs
    large_cutoff
        The cutoff radius for the large neighbour list

        Default is 9.0Å

    Returns
    -------
    list[AtomicGraph]
        List of atomic graphs for each molecule in ``molecules``, trimmed to ``model_cutoff``
    """
    # checks whether the cutoff is larger than the model cutoff
    if large_cutoff <= model_cutoff:
        raise ValueError(
            f"The large cutoff ({large_cutoff}) must be larger than the model cutoff ({model_cutoff})"
        )

    # warns if large cutoff is not significantly larger than model cutoff
    if large_cutoff - model_cutoff < 1.5:
        warn(
            f"The large cutoff ({large_cutoff}) is not significantly larger than the model cutoff ({model_cutoff}). This may lead to inefficient neighbour list calculations.",
        )

    graph = AtomicGraph.from_ase(starting_molecule, cutoff=large_cutoff)

    R_slices = [
        torch.tensor(atoms.positions, dtype=torch.float32)
        for atoms in molecules
    ]

    graphs = [
        trim_edges(graph._replace(R=R_slice), model_cutoff)
        for R_slice in R_slices
    ]

    return graphs


def get_nearest_neighbour_distance(atomic_graph: AtomicGraph) -> float:
    """
    Calculates the smallest nearest-neighbour distance in an atomic graph

    This is useful for determining the minimum distance between atoms throughout a hyperplane

    Parameters
    ----------
    atomic_graph
        The atomic graph to calculate the nearest-neighbour distance for

    Returns
    -------
    float
        The smallest nearest-neighbour distance in the atomic graph
    """
    positions = atomic_graph.R  # (num_atoms, 3)
    neighbour_list = atomic_graph.neighbour_list  # (num_edges, 2)

    # Extract connected atom indices
    i, j = neighbour_list[0, :], neighbour_list[1, :]

    # Compute distances
    distances = torch.norm(positions[i] - positions[j], dim=1)

    # Get smallest nearest-neighbour distance
    min_distance = torch.min(distances).item()

    return min_distance


def batch_calculate(
    model: GraphPESModel,
    graphs: list[AtomicGraph],
    batch_size: int = 10,
    get_forces: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch calculates the energies and (optionally) forces for each sample point
     
    Parses AtomicGraph objects to ``graph-pes`` GNN models

    Parameters
    ----------
    model
        The GraphPESModel to use for predictions
    graphs
        List of AtomicGraph objects to calculate energies and (optionally) forces for
    batch_size
        The batch size to use for model predictions

        Default is 10, which is a reasonable size for most models
    get_forces
        Whether to calculate forces in addition to energies

        Default is False, which only calculates energies

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Energies, forces, and absolute forces for each sample point
        - Energies: shape ``(num_points,)``
        - Forces: shape ``(num_points, num_atoms, 3)`` if get_forces is True
        - Absolute Forces: shape ``(num_points, num_atoms)`` if get_forces is True
    """
    dataloader = GraphDataLoader(graphs, batch_size=batch_size)
    energies = []
    forces = []
    abs_forces = np.array([])

    for batch in dataloader:
        energies.extend(model.predict_energy(batch).tolist())
        if get_forces:
            forces.extend(model.predict_forces(batch).tolist())

    if get_forces:
        f = np.array(forces).reshape(
            len(graphs), -1, 3
        )  # shape (num_points, num_atoms, 3)
        abs_forces = np.linalg.norm(f, axis=2)  # shape (num_points, num_atoms)

    return np.array(energies), np.array(forces), abs_forces