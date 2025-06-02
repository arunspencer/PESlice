import numpy as np
import ase
import torch
from graph_pes import GraphPESModel
from graph_pes.atomic_graph import AtomicGraph, trim_edges
from graph_pes.data import GraphDataLoader


def define_graphs(
    starting_molecule: ase.Atoms,
    molecules: list[ase.Atoms],
    model_cutoff: float,
    large_cutoff: float = 9.0,
) -> list[AtomicGraph]:
    """
    Creates atomic graphs for each structure using a shared large neighbuor list from
    the `starting_molecule`, trimming neighbours below `model_cutoff`.

    This avoids recomputing full neighbour lists for each structure.
    """

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
    Calculates the energies and (optionally) forces for each sample point in the hyperplane by parsing atomic graphs to the `graph-pes` GNN model
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