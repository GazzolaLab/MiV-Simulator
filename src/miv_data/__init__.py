from typing import Iterator

from neuroh5.io import (
    read_population_names,
    read_population_ranges,
    read_cell_attribute_info,
    scatter_read_trees,
    scatter_read_cell_attributes,
    scatter_read_graph,
)
from mpi4py import MPI
from miv_data import types


def read_cells_meta_data(
    filepath: str, comm: MPI.Comm = MPI.COMM_WORLD
) -> types.CellsMetaData:
    rank = comm.Get_rank()
    comm0 = comm.Split(int(rank == 0), 0)
    cell_attribute_info = None
    population_ranges = None
    population_names = None
    if rank == 0:
        population_names = read_population_names(filepath, comm0)
        (population_ranges, _) = read_population_ranges(filepath, comm0)
        cell_attribute_info = read_cell_attribute_info(
            filepath, population_names, comm=comm0
        )
    population_ranges = comm.bcast(population_ranges, root=0)
    population_names = comm.bcast(population_names, root=0)
    cell_attribute_info = comm.bcast(cell_attribute_info, root=0)

    comm0.Free()

    return types.CellsMetaData(
        population_names=population_names,
        population_ranges=population_ranges,
        cell_attribute_info=cell_attribute_info,
    )


def read_coordinates(
    filepath: str, population: str, comm: MPI.Comm = MPI.COMM_WORLD
) -> Iterator[tuple[int, tuple[float, float, float]]]:
    cell_attr_dict = scatter_read_cell_attributes(
        filepath,
        population,
        namespaces=["Generated Coordinates"],
        return_type="tuple",
        comm=comm,
    )
    coords_iter, coords_attr_info = cell_attr_dict["Generated Coordinates"]
    x_index = coords_attr_info.get("X Coordinate", None)
    y_index = coords_attr_info.get("Y Coordinate", None)
    z_index = coords_attr_info.get("Z Coordinate", None)
    for gid, cell_coords in coords_iter:
        yield gid, (
            cell_coords[x_index][0],
            cell_coords[y_index][0],
            cell_coords[z_index][0],
        )


def read_trees(
    filepath: str, population: str, comm: MPI.Comm = MPI.COMM_WORLD
) -> Iterator[tuple[int, types.Tree]]:
    (trees, forestSize) = scatter_read_trees(filepath, population, comm=comm)
    yield from trees


def read_projections(
    filepath: str,
    pre: types.PreSynapticPopulationName,
    post: types.PostSynapticPopulationName,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Iterator[tuple[int, tuple[list[int], types.Projection]]]:
    (graph, a) = scatter_read_graph(
        filepath,
        comm=comm,
        io_size=1,
        projections=[(pre, post)],
        namespaces=["Synapses", "Connections"],
    )

    yield from graph[post][pre]
