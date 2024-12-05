from typing import List, Optional, Tuple
import copy
import sys
import time
from collections import defaultdict
from mpi4py import MPI
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, writers
from miv_simulator import cells, spikedata, statedata, stimulus, synapses
from miv_simulator.volume import network_volume
from miv_simulator.env import Env
from miv_simulator.utils import (
    Struct,
    apply_filter,
    butter_bandpass_filter,
    get_module_logger,
    make_geometric_graph,
    signal_power_spectrogram,
    signal_psd,
    zip_longest,
    add_bins,
    update_bins,
    finalize_bins,
    get_low_pass_filtered_trace,
)
from miv_simulator.utils.neuron import h, interplocs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from neuroh5.io import (
    read_cell_attributes,
    read_population_names,
    read_population_ranges,
    NeuroH5ProjectionGen,
    bcast_cell_attributes,
)
from scipy import interpolate, ndimage, signal

if hasattr(h, "nrnmpi_init"):
    h.nrnmpi_init()

# This logger will inherit its settings from the root logger, created in miv_simulator.env
logger = get_module_logger(__name__)

# Default figure configuration
default_fig_options = Struct(
    figFormat="png",
    lw=2,
    figSize=(8, 8),
    fontSize=14,
    saveFig=None,
    showFig=True,
    colormap="jet",
    saveFigDir=None,
)

dflt_colors = [
    "#009BFF",
    "#E85EBE",
    "#00FF00",
    "#0000FF",
    "#FF0000",
    "#01FFFE",
    "#FFA6FE",
    "#FFDB66",
    "#006401",
    "#010067",
    "#95003A",
    "#007DB5",
    "#FF00F6",
    "#FFEEE8",
    "#774D00",
    "#90FB92",
    "#0076FF",
    "#D5FF00",
    "#FF937E",
    "#6A826C",
    "#FF029D",
    "#FE8900",
    "#7A4782",
    "#7E2DD2",
    "#85A900",
    "#FF0056",
    "#A42400",
    "#00AE7E",
    "#683D3B",
    "#BDC6FF",
    "#263400",
    "#BDD393",
    "#00B917",
    "#9E008E",
    "#001544",
    "#C28C9F",
    "#FF74A3",
    "#01D0FF",
    "#004754",
    "#E56FFE",
    "#788231",
    "#0E4MIV",
    "#91D0CB",
    "#BE9970",
    "#968AE8",
    "#BB8800",
    "#43002C",
    "#DEFF74",
    "#00FFC6",
    "#FFE502",
    "#620E00",
    "#008F9C",
    "#98FF52",
    "#7544B1",
    "#B500FF",
    "#00FF78",
    "#FF6E41",
    "#005F39",
    "#6B6882",
    "#5FAD4E",
    "#A75740",
    "#A5FFD2",
    "#FFB167",
]

rainbow_colors = [
    "#9400D3",
    "#4B0082",
    "#00FF00",
    "#FFFF00",
    "#FF7F00",
    "#FF0000",
]

raster_colors = [
    "#8dd3c7",
    "#ffed6f",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
]


def hex2rgb(hexcode):
    if hasattr(hexcode, "decode"):
        return tuple(float(b) / 255.0 for b in map(ord, hexcode[1:].decode("hex")))
    else:
        import codecs

        bhexcode = bytes(hexcode[1:], "utf-8")
        return tuple(float(b) / 255.0 for b in codecs.decode(bhexcode, "hex"))


def show_figure():
    plt.show()


def close_figure(fig):
    plt.close(fig)


def save_figure(file_name_prefix, fig=None, **kwargs):
    """

    :param file_name_prefix:
    :param fig: :class:'plt.Figure'
    :param kwargs: dict
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)
    fig_file_path = f"{file_name_prefix}.{fig_options.figFormat}"
    if fig_options.saveFigDir is not None:
        fig_file_path = f"{fig_options.saveFigDir}/{fig_file_path}"
    if fig is not None:
        fig.savefig(fig_file_path)
    else:
        plt.savefig(fig_file_path)


def plot_graph(
    x, y, z, start_idx, end_idx, edge_scalars=None, edge_color=None, **kwargs
):
    """
    Shows graph edges using Mayavi

    Parameters
    ----------
        x: ndarray
            x coordinates of the points
        y: ndarray
            y coordinates of the points
        z: ndarray
            z coordinates of the points
        edge_scalars: ndarray, optional
            optional data to give the color of the edges.
        kwargs:
            extra keyword arguments are passed to quiver3d.
    """
    from mayavi import mlab

    if edge_color is not None:
        kwargs["color"] = edge_color
    vec = mlab.quiver3d(
        x[start_idx],
        y[start_idx],
        z[start_idx],
        x[end_idx] - x[start_idx],
        y[end_idx] - y[start_idx],
        z[end_idx] - z[start_idx],
        scalars=edge_scalars,
        scale_factor=1,
        mode="2ddash",
        **kwargs,
    )
    b = mlab.points3d(x[0], y[0], z[0], mode="cone", scale_factor=3, **kwargs)
    if edge_scalars is not None:
        vec.glyph.color_mode = "color_by_scalar"
        cb = mlab.colorbar(vec, label_fmt="%.1f")
        cb.label_text_property.font_size = 14
    return vec


def clean_axes(axes, left=True, right=False):
    """
    Remove top and right axes from pyplot axes object.
    :param axes: list of pyplot.Axes
    :param top: bool
    :param left: bool
    :param right: bool
    """
    if type(axes) not in [np.ndarray, list]:
        axes = [axes]
    elif type(axes) == np.ndarray:
        axes = axes.flatten()
    for axis in axes:
        axis.tick_params(direction="out")
        axis.spines["top"].set_visible(False)
        if not right:
            axis.spines["right"].set_visible(False)
        if not left:
            axis.spines["left"].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()


def plot_spatial_bin_graph(graph_dict, **kwargs):
    import hiveplot as hv

    edge_dflt_colors = ["red", "crimson", "coral", "purple"]

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    label = graph_dict["label"]
    GU = graph_dict["U graph"]

    destination = graph_dict["destination"]
    sources = graph_dict["sources"]

    nodes = {}
    nodes[destination] = [(s, d) for s, d in GU.nodes() if s == destination]
    for source in sources:
        nodes[source] = [(s, d) for s, d in GU.nodes() if s == source]

    snodes = {}
    for group, nodelist in nodes.items():
        snodes[group] = sorted(nodelist)

    edges = {}
    for source in sources:
        edges[source] = [(u, v, d) for u, v, d in GU.edges(data=True) if v[0] == source]

    nodes_cmap = dict()
    nodes_cmap[destination] = "blue"
    for i, source in enumerate(sources):
        nodes_cmap[source] = raster_colors[i]

    edges_cmap = dict()
    for i, source in enumerate(sources):
        edges_cmap[source] = dflt_colors[i]

    hvpl = hv.HivePlot(snodes, edges, nodes_cmap, edges_cmap)
    hvpl.draw()

    filename = f"{label}.{fig_options.figFormat}"
    plt.savefig(filename)


def plot_coordinates(
    coords_path,
    population,
    namespace,
    index=0,
    graph_type="scatter",
    bin_size=0.01,
    xyz=False,
    **kwargs,
):
    """
    Plot coordinates

    :param coords_path:
    :param namespace:
    :param population:

    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    soma_coords = read_cell_attributes(coords_path, population, namespace=namespace)

    fig = plt.figure(1, figsize=plt.figaspect(1.0) * 2.0)
    ax = plt.gca()

    coord_U = {}
    coord_V = {}
    if xyz:
        for k, v in soma_coords:
            coord_U[k] = v["X Coordinate"][index]
            coord_V[k] = v["Y Coordinate"][index]
    else:
        for k, v in soma_coords:
            coord_U[k] = v["U Coordinate"][index]
            coord_V[k] = v["V Coordinate"][index]

    coord_U_array = np.asarray([coord_U[k] for k in sorted(coord_U.keys())])
    coord_V_array = np.asarray([coord_V[k] for k in sorted(coord_V.keys())])

    x_min = np.min(coord_U_array)
    x_max = np.max(coord_U_array)
    y_min = np.min(coord_V_array)
    y_max = np.max(coord_V_array)

    dx = int((x_max - x_min) / bin_size)
    dy = int((y_max - y_min) / bin_size)

    if graph_type == "scatter":
        ax.scatter(coord_U_array, coord_V_array, alpha=0.1, linewidth=0)
        ax.axis([x_min, x_max, y_min, y_max])
    elif graph_type == "histogram2d":
        (H, xedges, yedges) = np.histogram2d(
            coord_U_array, coord_V_array, bins=[dx, dy]
        )
        X, Y = np.meshgrid(xedges, yedges)
        Hint = H[:-1, :-1]
        levels = MaxNLocator(nbins=25).tick_values(Hint.min(), Hint.max())
        cmap = plt.get_cmap("jet")
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        p = ax.contourf(
            X[:-1, :-1] + (bin_size / 2),
            Y[:-1, :-1] + (bin_size / 2),
            H.T,
            levels=levels,
            cmap=cmap,
        )
        fig.colorbar(p, ax=ax, shrink=0.5, aspect=20)
    else:
        raise ValueError(f"Unknown graph type {graph_type}")

    if xyz:
        ax.set_xlabel("X coordinate (um)", fontsize=fig_options.fontSize)
        ax.set_ylabel("Y coordinate (um)", fontsize=fig_options.fontSize)
    else:
        ax.set_xlabel("U coordinate (septal - temporal)", fontsize=fig_options.fontSize)
        ax.set_ylabel(
            "V coordinate (supra - infrapyramidal)",
            fontsize=fig_options.fontSize,
        )

    ax.set_title(
        f"Coordinate distribution for population: {population}",
        fontsize=fig_options.fontSize,
    )

    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = f"{population} Coordinates.{fig_options.figFormat}"
            plt.savefig(filename)

    if fig_options.showFig:
        show_figure()

    return ax


# !needs refactoring
def plot_coords_in_volume(
    populations,
    coords_path,
    coords_namespace,
    config,
    scale=25.0,
    subpopulation=-1,
    subvol=False,
    verbose=False,
    mayavi=False,
    config_prefix="",
):
    from miv_simulator.geometry.geometry import get_total_extents

    env = Env(config=config, config_prefix=config_prefix)

    rotate = env.geometry["Parametric Surface"]["Rotation"]
    layer_extents = env.geometry["Parametric Surface"]["Layer Extents"]
    rotate = env.geometry["Parametric Surface"]["Rotation"]

    (extent_u, extent_v, extent_l) = get_total_extents(layer_extents)

    logger.info("Reading coordinates...")

    pop_min_extent = None
    pop_max_extent = None

    xcoords = []
    ycoords = []
    zcoords = []
    cmap = cm.get_cmap("Dark2")
    cmap_range = np.linspace(0, 1, num=len(populations))

    colors = []
    for pop_id, population in enumerate(populations):
        coords = read_cell_attributes(
            coords_path, population, namespace=coords_namespace
        )

        count = 0
        cxcoords = []
        cycoords = []
        czcoords = []
        for k, v in coords:
            count += 1
            cxcoords.append(v["X Coordinate"][0])
            cycoords.append(v["Y Coordinate"][0])
            czcoords.append(v["Z Coordinate"][0])
        if subpopulation > -1 and count > subpopulation:
            ridxs = np.random.choice(
                np.arange(len(cxcoords)), replace=False, size=subpopulation
            )
            cxcoords = list(np.asarray(cxcoords)[ridxs])
            cycoords = list(np.asarray(cycoords)[ridxs])
            czcoords = list(np.asarray(czcoords)[ridxs])

        colors += [cmap(cmap_range[pop_id]) for _ in range(len(cxcoords))]
        xcoords += cxcoords
        ycoords += cycoords
        zcoords += czcoords
        logger.info(f"Read {count} coordinates...")

        pop_distribution = env.geometry["Cell Distribution"][population]
        pop_layers = []
        for layer in pop_distribution:
            num_layer = pop_distribution[layer]
            if num_layer > 0:
                pop_layers.append(layer)

                if pop_min_extent is None:
                    pop_min_extent = np.asarray(layer_extents[layer][0])
                else:
                    pop_min_extent = np.minimum(
                        pop_min_extent, np.asarray(layer_extents[layer][0])
                    )

                if pop_max_extent is None:
                    pop_max_extent = np.asarray(layer_extents[layer][1])
                else:
                    pop_max_extent = np.maximum(
                        pop_min_extent, np.asarray(layer_extents[layer][1])
                    )

    pts = np.concatenate(
        (
            np.asarray(xcoords).reshape(-1, 1),
            np.asarray(ycoords).reshape(-1, 1),
            np.asarray(zcoords).reshape(-1, 1),
        ),
        axis=1,
    )

    if mayavi:
        from mayavi import mlab
    else:
        import matplotlib.pyplot as plt

    logger.info("Plotting coordinates...")
    if mayavi:
        mlab.points3d(*pts.T, color=(1, 1, 0), scale_factor=scale)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*pts.T, c=colors, s=int(scale))

    logger.info("Constructing volume...")
    from miv_simulator.volume import make_network_volume

    if subvol:
        subvol = make_network_volume(
            (pop_min_extent[0], pop_max_extent[0]),
            (pop_min_extent[1], pop_max_extent[1]),
            (pop_min_extent[2], pop_max_extent[2]),
            resolution=[3, 3, 3],
            rotate=rotate,
        )
    else:
        vol = make_network_volume(
            (extent_u[0], extent_u[1]),
            (extent_v[0], extent_v[1]),
            (extent_l[0], extent_l[1]),
            resolution=[3, 3, 3],
            rotate=rotate,
        )

    logger.info("Plotting volume...")

    if subvol:
        if mayavi:
            subvol.mplot_surface(color=(0, 0.4, 0), opacity=0.33)
        else:
            subvol.mplot_surface(color="k", alpha=0.33, figax=[fig, ax])
    else:
        if mayavi:
            vol.mplot_surface(color=(0, 1, 0), opacity=0.33)
        else:
            vol.mplot_surface(color="k", alpha=0.33, figax=[fig, ax])
    if mayavi:
        mlab.show()
    else:
        ax.view_init(-90, 0)
        plt.show()
        return fig


def plot_cell_tree(
    population,
    gid,
    tree_dict,
    line_width=1.0,
    sample=0.05,
    color_edge_scalars=True,
    mst=False,
    conn_loc=True,
    mayavi=False,
    **kwargs,
):
    import networkx as nx

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    xcoords = tree_dict["x"]
    ycoords = tree_dict["y"]
    zcoords = tree_dict["z"]
    swc_type = tree_dict["swc_type"]
    layer = tree_dict["layer"]
    secnodes = tree_dict["section_topology"]["nodes"]
    src = tree_dict["section_topology"]["src"]
    dst = tree_dict["section_topology"]["dst"]
    loc = tree_dict["section_topology"]["loc"]

    x = xcoords.reshape(
        -1,
    )
    y = ycoords.reshape(
        -1,
    )
    z = zcoords.reshape(
        -1,
    )

    edges = []
    for sec, nodes in secnodes.items():
        for i in range(1, len(nodes)):
            srcnode = nodes[i - 1]
            dstnode = nodes[i]
            edges.append((srcnode, dstnode))

    loc_x = []
    loc_y = []
    loc_z = []
    for s, d, l in zip(src, dst, loc):
        srcnode = secnodes[s][l]
        dstnode = secnodes[d][0]
        edges.append((srcnode, dstnode))
        loc_x.append(x[srcnode])
        loc_y.append(y[srcnode])
        loc_z.append(z[srcnode])

    conn_loc_x = np.asarray(loc_x, dtype=np.float64)
    conn_loc_y = np.asarray(loc_y, dtype=np.float64)
    conn_loc_z = np.asarray(loc_z, dtype=np.float64)

    # Make a NetworkX graph out of our point and edge data
    g = make_geometric_graph(x, y, z, edges)

    edges = g.edges
    # Compute minimum spanning tree using networkx
    # nx.mst returns an edge generator
    if mst:
        edges = nx.minimum_spanning_tree(g).edges(data=True)

    edge_array = np.array(list(edges)).T
    start_idx = edge_array[0, :]
    end_idx = edge_array[1, :]

    start_idx = start_idx.astype(np.int_)
    end_idx = end_idx.astype(np.int_)
    if color_edge_scalars:
        edge_scalars = z[start_idx]
        edge_color = None
    else:
        edge_scalars = None
        edge_color = hex2rgb(rainbow_colors[gid % len(rainbow_colors)])

    if mayavi:
        from mayavi import mlab

        mlab.figure(bgcolor=(0, 0, 0))
        fig = mlab.gcf()

        # Plot this with Mayavi
        g = plot_graph(
            x,
            y,
            z,
            start_idx,
            end_idx,
            edge_scalars=edge_scalars,
            edge_color=edge_color,
            opacity=0.8,
            colormap="summer",
            line_width=line_width,
            figure=fig,
        )

        if conn_loc:
            conn_pts = mlab.points3d(
                conn_loc_x,
                conn_loc_y,
                conn_loc_z,
                figure=fig,
                mode="2dcross",
                colormap="copper",
                scale_factor=10,
            )

        fig.scene.x_plus_view()
        if fig_options.saveFig:
            mlab.savefig(
                f"{population}_{gid}_cell_tree.x3d",
                figure=fig,
                magnification=10,
            )
        if fig_options.showFig:
            mlab.show()

    else:
        fig = plt.figure(figsize=fig_options.figSize)
        ax = fig.add_subplot(projection="3d")

        layer_set = set(layer)
        sct = ax.scatter(
            x,
            y,
            zs=z,
            c=layer,
            alpha=0.7,
        )
        # produce a legend with the unique colors from the scatter
        legend_elements = sct.legend_elements()
        layer_legend = ax.legend(*legend_elements, loc="upper right", title="Layer")
        ax.add_artist(layer_legend)

        for i, j in g.edges:
            e_x = (x[i], x[j])
            e_y = (y[i], y[j])
            e_z = (z[i], z[j])

            ax.plot(e_x, e_y, e_z, c="black", alpha=0.5)
            ax.view_init(30)
            ax.set_axis_off

        if fig_options.saveFig:
            if isinstance(fig_options.saveFig, str):
                filename = fig_options.saveFig
            else:
                filename = f"{population}_{gid}_cell_tree.{fig_options.figFormat}"
            plt.savefig(filename)
            print(f"Save figure: {filename}")

        if fig_options.showFig:
            show_figure()

    # return fig


## Plot spike raster
def plot_spike_raster(
    input_path,
    namespace_id,
    include=["eachPop"],
    time_range=None,
    time_variable="t",
    max_spikes=int(1e6),
    labels="legend",
    pop_rates=True,
    spike_hist=None,
    spike_hist_bin=5,
    include_artificial=True,
    marker=".",
    **kwargs,
):
    """
    Raster plot of network spike times. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    max_spikes (int): maximum number of spikes that will be plotted  (default: 1e6)
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    pop_rates = (True|False): Include population rates (default: False)
    spike_hist (None|'overlay'|'subplot'): overlay line over raster showing spike histogram (spikes/bin) (default: False)
    spike_hist_bin (int): Size of bin in ms to use for histogram (default: 5)
    marker (char): Marker for each spike (default: '|')
    """

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    mpl.rcParams["font.size"] = fig_options.fontSize

    (population_ranges, N) = read_population_ranges(input_path)
    population_names = read_population_names(input_path)

    total_num_cells = 0
    pop_num_cells = {}
    pop_start_inds = {}
    for k in population_names:
        pop_start_inds[k] = population_ranges[k][0]
        pop_num_cells[k] = population_ranges[k][1]
        total_num_cells += population_ranges[k][1]

    include = list(include)
    # Replace 'eachPop' with list of populations
    if "eachPop" in include:
        include.remove("eachPop")
        for pop in population_names:
            include.append(pop)

    # sort according to start index
    include.sort(key=lambda x: pop_start_inds[x])

    spkdata = spikedata.read_spike_events(
        input_path,
        include,
        namespace_id,
        include_artificial=include_artificial,
        spike_train_attr_name=time_variable,
        time_range=time_range,
    )

    spkpoplst = spkdata["spkpoplst"]
    spkindlst = spkdata["spkindlst"]
    spktlst = spkdata["spktlst"]
    num_cell_spks = spkdata["num_cell_spks"]
    pop_active_cells = spkdata["pop_active_cells"]
    tmin = spkdata["tmin"]
    tmax = spkdata["tmax"]
    fraction_active = {
        pop_name: float(len(pop_active_cells[pop_name]))
        / float(pop_num_cells[pop_name])
        for pop_name in include
    }

    time_range = [tmin, tmax]

    # Calculate spike histogram if requested
    if spike_hist:
        all_spkts = []
        sphist_x = None
        sphist_y = None
        bin_edges = None
        if len(spktlst) > 0:
            all_spkts = np.concatenate([np.concatenate(lst, axis=0) for lst in spktlst])
            sphist_y, bin_edges = np.histogram(
                all_spkts,
                bins=np.arange(time_range[0], time_range[1], spike_hist_bin),
            )
            sphist_x = bin_edges[:-1] + (spike_hist_bin / 2)
        else:
            spike_hist = None

    maxN = 0
    minN = N

    avg_rates = {}
    tsecs = (time_range[1] - time_range[0]) / 1e3
    for i, pop_name in enumerate(spkpoplst):
        pop_num = len(pop_active_cells[pop_name])
        maxN = max(maxN, max(pop_active_cells[pop_name]))
        minN = min(minN, min(pop_active_cells[pop_name]))
        if pop_num > 0:
            if num_cell_spks[pop_name] == 0:
                avg_rates[pop_name] = 0
            else:
                avg_rates[pop_name] = (num_cell_spks[pop_name] / pop_num) / tsecs

    pop_colors = {
        pop_name: dflt_colors[ipop % len(raster_colors)]
        for ipop, pop_name in enumerate(spkpoplst)
    }

    pop_spk_dict = {
        pop_name: (pop_spkinds, pop_spkts)
        for (pop_name, pop_spkinds, pop_spkts) in zip(spkpoplst, spkindlst, spktlst)
    }

    n_subplots = 1
    if spike_hist is None:
        n_subplots = max(len(spkpoplst), 1)
        fig, axes = plt.subplots(
            nrows=n_subplots, sharex=True, figsize=fig_options.figSize
        )
    elif spike_hist == "subplot":
        n_subplots = max(len(spkpoplst), 1) + 1
        fig, axes = plt.subplots(
            nrows=n_subplots,
            sharex=True,
            figsize=fig_options.figSize,
            gridspec_kw={"height_ratios": [1] * len(spkpoplst) + [2]},
        )
    fig.suptitle("Spike Raster", fontsize=fig_options.fontSize)

    sctplots = []

    if n_subplots == 1:
        axes = [axes]

    for i, pop_name in enumerate(spkpoplst):
        if pop_name not in pop_spk_dict:
            continue

        pop_spkinds, pop_spkts = pop_spk_dict[pop_name]

        logger.info(
            f"population {pop_name}: spike counts: {np.unique(pop_spkinds, return_counts=True)}"
        )

        if max_spikes is not None:
            if int(max_spikes) < len(pop_spkinds):
                logger.info(
                    f"Loading only randomly sampled {max_spikes} out of {len(pop_spkts)} spikes for population {pop_name}"
                )
                sample_inds = np.random.randint(
                    0, len(pop_spkinds) - 1, size=int(max_spikes)
                )
                pop_spkts = pop_spkts[sample_inds]
                pop_spkinds = pop_spkinds[sample_inds]

        sct = None
        if len(pop_spkinds) > 0:
            for this_pop_spkts, this_pop_spkinds in zip(pop_spkts, pop_spkinds):
                sct = axes[i].scatter(
                    this_pop_spkts,
                    this_pop_spkinds,
                    s=1,
                    linewidths=fig_options.lw,
                    marker=marker,
                    c=pop_colors[pop_name],
                    alpha=0.5,
                    label=pop_name,
                )

        axes[i].spines["top"].set_visible(False)
        axes[i].spines["bottom"].set_visible(False)
        axes[i].spines["left"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        sctplots.append(sct)

        N = pop_num_cells[pop_name]
        S = pop_start_inds[pop_name]
        axes[i].set_ylim(S, S + N - 1)

    lgd_info = [
        (
            100.0 * fraction_active.get(pop_name, 0.0),
            avg_rates.get(pop_name, 0.0),
        )
        for pop_name in spkpoplst
    ]

    # set raster plot y tick labels to the middle of the index range for each population
    for pop_name, a in zip_longest(spkpoplst, fig.axes):
        if pop_name not in pop_active_cells:
            continue
        if a is None:
            continue
        if len(pop_active_cells[pop_name]) > 0:
            maxN = max(pop_active_cells[pop_name])
            minN = min(pop_active_cells[pop_name])
            loc = pop_start_inds[pop_name] + 0.5 * (maxN - minN)
            yaxis = a.get_yaxis()
            yaxis.set_ticks([loc])
            yaxis.set_ticklabels([pop_name])
            yaxis.set_tick_params(length=0)
            a.get_xaxis().set_tick_params(length=0)

    if spike_hist:
        # Plot spike histogram
        pch = interpolate.pchip(sphist_x, sphist_y)
        res_npts = int(sphist_x.max() - sphist_x.min())
        sphist_x_res = np.linspace(
            sphist_x.min(), sphist_x.max(), res_npts, endpoint=True
        )
        sphist_y_res = pch(sphist_x_res)

        if spike_hist == "overlay":
            ax2 = axes[-1].twinx()
            ax2.plot(sphist_x_res, sphist_y_res, linewidth=0.5)
            ax2.set_ylabel(
                "Spike count", fontsize=fig_options.fontSize
            )  # add yaxis label in opposite side
            ax2.set_xlim(time_range)
        elif spike_hist == "subplot":
            ax2 = axes[-1]
            ax2.bar(sphist_x_res, sphist_y_res, linewidth=1.0)
            ax2.set_xlabel("Time (ms)", fontsize=fig_options.fontSize)
            ax2.set_ylabel("Spikes", fontsize=fig_options.fontSize)
            ax2.set_xlim(time_range)

    #    locator=MaxNLocator(prune='both', nbins=10)
    #    ax2.xaxis.set_major_locator(locator)

    if labels == "legend":
        # Shrink axes by 15%
        if n_subplots > 1:
            for ax in axes:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        if pop_rates:
            lgd_labels = [
                f"{pop_name} ({info[0]:.02f}% active; {info[1]:.3g} Hz)"
                for pop_name, info in zip_longest(spkpoplst, lgd_info)
            ]
        else:
            lgd_labels = [
                f"{pop_name} ({info[0]:.02f}% active)"
                for pop_name, info in zip_longest(spkpoplst, lgd_info)
            ]
        # Add legend
        lgd = fig.legend(
            sctplots,
            lgd_labels,
            loc="center right",
            fontsize="small",
            scatterpoints=1,
            markerscale=5.0,
            bbox_to_anchor=(1.002, 0.5),
            bbox_transform=plt.gcf().transFigure,
        )
        fig.artists.append(lgd)

    elif labels == "overlay":
        if pop_rates:
            lgd_labels = [
                f"{pop_name} ({info[0]:.02f}% active; {info[1]:.3g} Hz)"
                for pop_name, info in zip_longest(spkpoplst, lgd_info)
            ]
        else:
            lgd_labels = [
                f"{pop_name} ({info[0]:.02f}% active)"
                for pop_name, info in zip_longest(spkpoplst, lgd_info)
            ]
        for i, (pop_name, lgd_label) in enumerate(
            zip_longest(spkpoplst, lgd_labels)
        ):
            at = AnchoredText(
                f"{pop_name} {lgd_label}",
                loc="upper right",
                borderpad=0.01,
                prop=dict(size=fig_options.fontSize),
            )
            axes[i].add_artist(at)
        max_label_len = max(len(l) for l in lgd_labels)

    elif labels == "yticks":
        for pop_name, info, a in zip_longest(spkpoplst, lgd_info, fig.axes):
            if a is None or info is None:
                continue

            if pop_rates:
                label = f"\n({info[0]:.02f}% active;\n {info[1]:.3g} Hz)"
            else:
                label = f"\n({info[0]:.02f}% active)"

            maxN = max(pop_active_cells[pop_name])
            minN = min(pop_active_cells[pop_name])
            loc = pop_start_inds[pop_name] + 0.5 * (maxN - minN)
            a.tick_params(axis="y", labelsize="x-small")
            a.set_yticks([loc, loc])
            a.set_yticklabels([pop_name, label])
            yticklabels = a.get_yticklabels()
            # Create offset transform in x direction
            dx = -80 / 72.0
            dy = 0 / 72.0
            offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            # apply offset transform to labels.
            yticklabels[0].set_transform(yticklabels[0].get_transform() + offset)
            dx = -80 / 72.0
            dy = 0 / 72.0
            offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            yticklabels[1].set_ha("left")
            yticklabels[1].set_transform(yticklabels[1].get_transform() + offset)

    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # save figure
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = f"{namespace_id} raster.{fig_options.figFormat}"
            plt.savefig(filename)

    # show fig
    if fig_options.showFig:
        show_figure()

    return fig


def plot_spike_histogram(
    input_path: str,
    namespace_id,
    config_path: Optional[str] = None,
    include: List[str] = ["eachPop"],  # TODO: Probably not safe
    time_variable: str = "t",
    time_range: Optional[Tuple[int, int]] = None,
    pop_rates: bool = False,
    bin_size: int = 5,
    smooth: float = 0,
    quantity: str = "rate",
    include_artificial: bool = True,
    progress: bool = False,
    overlay: bool = True,
    graph_type: str = "bar",
    **kwargs,
):
    """
    Plots spike histogram. Returns figure handle.

    Parameters
    ----------
    input_path : str
        file with spike data
    namespace_id :
        attribute namespace for spike events
    config_path : Optional[str]
        config_path
    include : List[str] (["eachPop", <population name>])
        List of data series to include. (default: ["eachPop"] expands to the name of each population)
    time_variable : str
        Name of variable containing spike times (default: "t")
    time_range : Optional[Tuple[int,int]] ([start:stop])
        Time range of spikes shown. If None shows all. (default: None)
    pop_rates : bool
        pop_rates
    bin_size : int
        Size in ms of each bin (default: 5)
    smooth : float
        smooth
    quantity : str ("rate", "count")
        Quantity of y axis (firing rate in Hz, or spike count) (default: "rate")
    include_artificial : bool
        include_artificial
    progress : bool
        progress
    overlay : bool
        Whether to overlay the data lines or plot in separate subplots (default: True)
    graph_type : str ("line", "bar")
        Type of graph to use (line graph or bar plot) (default: "line")
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    baks_config = copy.copy(kwargs)

    env = None
    if config_path is not None:
        env = Env(config=config_path)
        if env.analysis_config is not None:
            baks_config.update(env.analysis_config["Firing Rate Inference"])

    (population_ranges, N) = read_population_ranges(input_path)
    population_names = read_population_names(input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    # Replace 'eachPop' with list of populations
    if "eachPop" in include:
        include.remove("eachPop")
        for pop in population_names:
            include.append(pop)
        include.reverse()

    spkdata = spikedata.read_spike_events(
        input_path,
        include,
        namespace_id,
        spike_train_attr_name=time_variable,
        time_range=time_range,
        include_artificial=include_artificial,
    )

    spkpoplst = spkdata["spkpoplst"]
    spkindlst = spkdata["spkindlst"]
    spktlst = spkdata["spktlst"]
    num_cell_spks = spkdata["num_cell_spks"]
    pop_active_cells = spkdata["pop_active_cells"]
    tmin = spkdata["tmin"]
    tmax = spkdata["tmax"]

    time_range = [tmin, tmax]

    avg_rates = {}
    maxN = 0
    minN = N
    if pop_rates:
        tsecs = (time_range[1] - time_range[0]) / 1e3
        for i, pop_name in enumerate(spkpoplst):
            pop_num = len(pop_active_cells[pop_name])
            maxN = max(maxN, max(pop_active_cells[pop_name]))
            minN = min(minN, min(pop_active_cells[pop_name]))
            if pop_num > 0:
                if num_cell_spks[pop_name] == 0:
                    avg_rates[pop_name] = 0
                else:
                    avg_rates[pop_name] = (num_cell_spks[pop_name] / pop_num) / tsecs

    # Y-axis label
    if quantity == "rate":
        yaxisLabel = "Mean cell firing rate (Hz)"
    elif quantity == "count":
        yaxisLabel = "Spike count"
    elif quantity == "active":
        yaxisLabel = "Active cell count"
    else:
        logger.error(f"Invalid quantity value {quantity}")
        return

    # create fig
    fig, axes = plt.subplots(
        len(spkpoplst), 1, figsize=fig_options.figSize, sharex=True
    )

    time_bins = np.arange(time_range[0], time_range[1], bin_size)

    hist_dict = {}
    if quantity == "rate":
        for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
            spkdict = spikedata.make_spike_dict(spkinds, spkts)
            sdf_dict = spikedata.spike_density_estimate(
                subset, spkdict, time_bins, progress=progress, **baks_config
            )
            bin_dict = defaultdict(lambda: {"rates": 0.0, "active": 0})
            for ind, dct in sdf_dict.items():
                rate = dct["rate"]
                for ibin in range(0, len(time_bins)):
                    d = bin_dict[ibin]
                    bin_rate = rate[ibin]
                    d["rates"] += bin_rate
                    d["active"] += 1
            hist_dict[subset] = bin_dict
            logger.info(
                "Calculated spike rates for %i cells in population %s"
                % (len(sdf_dict), subset)
            )
    else:
        for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
            spkdict = spikedata.make_spike_dict(spkinds, spkts)
            count_bin_dict = spikedata.spike_bin_counts(spkdict, time_bins)
            bin_dict = defaultdict(lambda: {"counts": 0, "active": 0})
            for ind, counts in count_bin_dict.items():
                for ibin in range(0, len(time_bins) - 1):
                    d = bin_dict[ibin]
                    d["counts"] += counts[ibin]
                    d["active"] += 1
            hist_dict[subset] = bin_dict
            logger.info(
                "Calculated spike counts for %i cells in population %s"
                % (len(count_bin_dict), subset)
            )

    del spkindlst, spktlst

    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):
        hist_x = time_bins + (bin_size / 2)
        bin_dict = hist_dict[subset]

        if quantity == "rate":
            hist_y = np.asarray(
                [
                    (
                        (bin_dict[ibin]["rates"] / bin_dict[ibin]["active"])
                        if bin_dict[ibin]["active"] > 0
                        else 0.0
                    )
                    for ibin in range(0, len(time_bins))
                ]
            )
        elif quantity == "active":
            hist_y = np.asarray(
                [bin_dict[ibin]["active"] for ibin in range(0, len(time_bins))]
            )
        else:
            hist_y = np.asarray(
                [bin_dict[ibin]["counts"] for ibin in range(0, len(time_bins))]
            )

        del bin_dict
        del hist_dict[subset]

        color = dflt_colors[iplot % len(dflt_colors)]

        if pop_rates:
            label = str(subset) + " (%i active; %.3g Hz)" % (
                len(pop_active_cells[subset]),
                avg_rates[subset],
            )
        else:
            label = str(subset) + f" ({len(pop_active_cells[subset])} active)"

        ax = plt.subplot(len(spkpoplst), 1, (iplot + 1))
        plt.title(label, fontsize=fig_options.fontSize)
        ax.tick_params(labelsize=fig_options.fontSize)
        if iplot < len(spkpoplst) - 1:
            ax.xaxis.set_visible(False)

        if smooth:
            hsignal = signal.savgol_filter(
                hist_y,
                window_length=2 * (len(hist_y) / 16) + 1,
                polyorder=smooth,
            )
        else:
            hsignal = hist_y

        if graph_type == "line":
            ax.plot(hist_x, hsignal, linewidth=fig_options.lw, color=color)
        elif graph_type == "bar":
            ax.bar(
                hist_x,
                hsignal,
                width=bin_size,
                color=color,
                edgecolor="black",
                alpha=0.85,
            )

        if iplot == 0:
            ax.set_ylabel(yaxisLabel, fontsize=fig_options.fontSize)
        if iplot == len(spkpoplst) - 1:
            ax.set_xlabel("Time (ms)", fontsize=fig_options.fontSize)
        else:
            ax.tick_params(labelbottom="off")

        ax.set_xlim(time_range)

    plt.tight_layout()

    # Add legend
    if overlay:
        for i, subset in enumerate(spkpoplst):
            plt.plot(0, 0, color=dflt_colors[i % len(dflt_colors)], label=str(subset))
        plt.legend(
            fontsize=fig_options.fontSize,
            bbox_to_anchor=(1.04, 1),
            loc=2,
            borderaxespad=0.0,
        )
        maxLabelLen = min(10, max(len(str(l)) for l in include))
        plt.subplots_adjust(right=(0.9 - 0.012 * maxLabelLen))

    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = namespace_id + " " + f"histogram.{fig_options.figFormat}"
        plt.savefig(filename)

    if fig_options.showFig:
        show_figure()

    return fig


def plot_lfp(
    input_path,
    config_path=None,
    time_range=None,
    compute_psd=False,
    window_size=4096,
    frequency_range=(0, 400.0),
    overlap=0.9,
    bandpass_filter=False,
    dt=None,
    **kwargs,
):
    """
    Line plot of LFP state variable (default: v). Returns figure handle.

    config: path to model configuration file
    input_path: file with LFP trace data
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    env = None
    if config_path is not None:
        env = Env(config=config_path)

    nrows = 1
    if env is not None:
        nrows = len(env.LFP_config)
    ncols = 1
    psd_col = 1
    if compute_psd:
        ncols += 1

    gs = gridspec.GridSpec(nrows, ncols, width_ratios=[3, 1] if ncols > 1 else [1])
    fig = plt.figure(figsize=fig_options.figSize)
    if env is None:
        lfp_array = np.loadtxt(
            input_path, dtype=np.dtype([("t", np.float32), ("v", np.float32)])
        )

        if time_range is None:
            t = lfp_array["t"]
            v = lfp_array["v"]
        else:
            tlst = []
            vlst = []
            for t, v in zip(lfp_array["t"], lfp_array["v"]):
                if time_range[0] <= t <= time_range[1]:
                    tlst.append(t)
                    vlst.append(v)
            t = np.asarray(tlst)
            v = np.asarray(vlst)

        if dt is None:
            raise RuntimeError("plot_lfp: dt must be provided when config_path is None")
        Fs = 1000.0 / dt

        if compute_psd:
            psd, freqs, peak_index = signal_psd(
                v,
                frequency_range=frequency_range,
                Fs=Fs,
                window_size=window_size,
                overlap=overlap,
            )

        filtered_v = None
        if bandpass_filter:
            filtered_v = apply_filter(
                v,
                butter_bandpass_filter(
                    max(frequency_range[0], 1.0),
                    frequency_range[1],
                    Fs,
                    order=2,
                ),
            )

        iplot = 0
        ax = plt.subplot(gs[iplot, 0])
        ax.set_title("LFP", fontsize=fig_options.fontSize)
        ax.plot(t, v, linewidth=fig_options.lw)
        ax.set_xlabel("Time (ms)", fontsize=fig_options.fontSize)
        ax.set_ylabel("Field Potential (mV)", fontsize=fig_options.fontSize)

        if bandpass_filter:
            if filtered_v is not None:
                ax.plot(
                    t,
                    filtered_v,
                    label="Filtered LFP",
                    color="red",
                    linewidth=fig_options.lw,
                )
        if compute_psd:
            ax = plt.subplot(gs[iplot, psd_col])
            ax.plot(freqs, psd, linewidth=fig_options.lw)
            ax.set_xlabel("Frequency (Hz)", fontsize=fig_options.fontSize)
            ax.set_ylabel(
                "Power Spectral Density (dB/Hz)", fontsize=fig_options.fontSize
            )
            ax.set_title(
                f"PSD (peak: {freqs[peak_index]:.3f} Hz)",
                fontsize=fig_options.fontSize,
            )

        if fig_options.saveFig:
            if isinstance(fig_options.saveFig, str):
                filename = fig_options.saveFig
            else:
                filename = f"MIV LFP.{fig_options.figFormat}"
                plt.savefig(filename)

        # show fig
        if fig_options.showFig:
            show_figure()

    else:
        for iplot, (lfp_label, lfp_config_dict) in enumerate(env.LFP_config.items()):
            namespace_id = f"Local Field Potential {str(lfp_label)}"
            import h5py

            infile = h5py.File(input_path)

            logger.info(f"plot_lfp: reading data for {namespace_id}...")
            if time_range is None:
                t = infile[namespace_id]["t"]
                v = infile[namespace_id]["v"]
                t = np.asarray(t)
                v = np.asarray(v)
            else:
                tlst = []
                vlst = []
                for t, v in zip(infile[namespace_id]["t"], infile[namespace_id]["v"]):
                    if time_range[0] <= t <= time_range[1]:
                        tlst.append(t)
                        vlst.append(v)
                t = np.asarray(tlst)
                v = np.asarray(vlst)

            dt = lfp_config_dict["dt"]
            Fs = 1000.0 / dt

            if compute_psd:
                psd, freqs, peak_index = signal_psd(
                    v,
                    Fs=Fs,
                    frequency_range=frequency_range,
                    window_size=window_size,
                    overlap=overlap,
                )

            filtered_v = None
            if bandpass_filter:
                filtered_v = apply_filter(
                    v,
                    butter_bandpass_filter(
                        max(frequency_range[0], 1.0),
                        frequency_range[1],
                        Fs,
                        order=2,
                    ),
                )

            ax = plt.subplot(gs[iplot, 0])
            ax.set_title(f"{namespace_id}", fontsize=fig_options.fontSize)
            ax.plot(t, v, label=lfp_label, linewidth=fig_options.lw)
            ax.set_xlabel("Time (ms)", fontsize=fig_options.fontSize)
            ax.set_ylabel("Field Potential (mV)", fontsize=fig_options.fontSize)
            if bandpass_filter:
                if filtered_v is not None:
                    ax.plot(
                        t,
                        filtered_v,
                        label=f"{lfp_label} (filtered)",
                        color="red",
                        linewidth=fig_options.lw,
                    )
            if compute_psd:
                ax = plt.subplot(gs[iplot, psd_col])
                ax.plot(freqs, psd, linewidth=fig_options.lw)
                ax.set_xlabel("Frequency (Hz)", fontsize=fig_options.fontSize)
                ax.set_ylabel(
                    "Power Spectral Density (dB/Hz)",
                    fontsize=fig_options.fontSize,
                )
                ax.set_title(
                    f"PSD (peak: {freqs[peak_index]:.3f} Hz)",
                    fontsize=fig_options.fontSize,
                )

        # save figure
        if fig_options.saveFig:
            if isinstance(fig_options.saveFig, str):
                filename = fig_options.saveFig
            else:
                filename = namespace_id + f".{fig_options.figFormat}"
                plt.savefig(filename)

        # show fig
        if fig_options.showFig:
            show_figure()

    return fig


def plot_lfp_spectrogram(
    input_path,
    config_path=None,
    time_range=None,
    window_size=4096,
    overlap=0.9,
    frequency_range=(0, 400.0),
    dt=None,
    **kwargs,
):
    """
    Line plot of LFP power spectrogram. Returns figure handle.

    config: path to model configuration file
    input_path: file with LFP trace data
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    mpl.rcParams["font.size"] = fig_options.fontSize

    env = None
    if config_path is not None:
        env = Env(config=config_path)

    nrows = 1
    if env is not None:
        nrows = len(env.LFP_config)

    ncols = 1
    gs = gridspec.GridSpec(nrows, ncols, width_ratios=[3, 1] if ncols > 1 else [1])
    fig = plt.figure(figsize=fig_options.figSize)
    if env is None:
        lfp_array = np.loadtxt(
            input_path, dtype=np.dtype([("t", np.float32), ("v", np.float32)])
        )

        if time_range is None:
            t = lfp_array["t"]
            v = lfp_array["v"]
        else:
            tlst = []
            vlst = []
            for t, v in zip(lfp_array["t"], lfp_array["v"]):
                if time_range[0] <= t <= time_range[1]:
                    tlst.append(t)
                    vlst.append(v)
            t = np.asarray(tlst)
            v = np.asarray(vlst)

        if dt is None:
            raise RuntimeError(
                "plot_lfp_spectrogram: dt must be provided when config_path is None"
            )
        Fs = int(1000.0 / dt)

        freqs, t, Sxx = signal_power_spectrogram(v, Fs, window_size, overlap)
        freqinds = np.where(
            (freqs >= frequency_range[0]) & (freqs <= frequency_range[1])
        )

        freqs = freqs[freqinds]
        sxx = Sxx[freqinds, :][0]

        iplot = 0
        ax = plt.subplot(gs[iplot, 0])
        ax.set_xlim([0.4, 0.8])
        ax.set_ylim(*frequency_range)
        ax.set_title("LFP Spectrogram", fontsize=fig_options.fontSize)
        pcm = ax.pcolormesh(t, freqs, sxx, cmap="jet")
        ax.set_xlabel("Time (s)", fontsize=fig_options.fontSize)
        ax.set_ylabel("Frequency (Hz)", fontsize=fig_options.fontSize)
        ax.tick_params(axis="both", labelsize=fig_options.fontSize)
        fig.colorbar(pcm, ax=ax)

        # save figure
        if fig_options.saveFig:
            if isinstance(fig_options.saveFig, str):
                filename = fig_options.saveFig
            else:
                filename = namespace_id + f".{fig_options.figFormat}"
            plt.savefig(filename)

        # show fig
        if fig_options.showFig:
            show_figure()

    else:
        for iplot, (lfp_label, lfp_config_dict) in enumerate(env.LFP_config.items()):
            namespace_id = f"Local Field Potential {str(lfp_label)}"
            import h5py

            infile = h5py.File(input_path)

            logger.info(f"plot_lfp: reading data for {namespace_id}...")
            if time_range is None:
                t = infile[namespace_id]["t"]
                v = infile[namespace_id]["v"]
            else:
                tlst = []
                vlst = []
                for t, v in zip(infile[namespace_id]["t"], infile[namespace_id]["v"]):
                    if time_range[0] <= t <= time_range[1]:
                        tlst.append(t)
                        vlst.append(v)
                t = np.asarray(tlst)
                v = np.asarray(vlst)

            dt = lfp_config_dict["dt"]

            Fs = int(1000.0 / dt)

            freqs, t, Sxx = signal_power_spectrogram(v, Fs, window_size, overlap)
            freqinds = np.where(
                (freqs >= frequency_range[0]) & (freqs <= frequency_range[1])
            )

            freqs = freqs[freqinds]
            sxx = Sxx[freqinds, :][0]

            ax = plt.subplot(gs[iplot, 0])

            ax.set_ylim(*frequency_range)
            ax.set_title(f"{namespace_id}", fontsize=fig_options.fontSize)
            ax.pcolormesh(t, freqs, sxx, cmap="jet")
            ax.set_xlabel("Time (s)", fontsize=fig_options.fontSize)
            ax.set_ylabel("Frequency (Hz)", fontsize=fig_options.fontSize)

        # save figure
        if fig_options.saveFig:
            if isinstance(fig_options.saveFig, str):
                filename = fig_options.saveFig
            else:
                filename = namespace_id + f".{fig_options.figFormat}"
            plt.savefig(filename)

        # show fig
        if fig_options.showFig:
            show_figure()

    return fig


## Plot biophys cell tree
def plot_biophys_cell_tree(
    env,
    biophys_cell,
    node_filters={"swc_types": ["apical", "basal"]},
    plot_synapses=False,
    synapse_filters=None,
    syn_source_threshold=0.0,
    line_width=8.0,
    plot_method="neuron",
    **kwargs,
):
    """
    Plot cell morphology and optionally synapse locations.

    """

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    morph_graph = cells.make_morph_graph(biophys_cell, node_filters=node_filters)

    gid = biophys_cell.gid
    population = biophys_cell.population_name

    # Obtain synapse xyz locations
    syn_attrs = env.synapse_attributes
    synapse_filters = synapses.get_syn_filter_dict(env, synapse_filters, convert=True)
    syns_dict = syn_attrs.filter_synapses(biophys_cell.gid, **synapse_filters)
    syn_sec_dict = defaultdict(list)
    if (syn_source_threshold is not None) and (syn_source_threshold > 0.0):
        syn_source_count = defaultdict(int)
        for syn_id, syn in syns_dict.items():
            syn_source_count[syn.source.gid] += 1
        syn_source_max = 0
        syn_source_pctile = {}
        for source_id, source_id_count in syn_source_count.items():
            syn_source_max = max(syn_source_max, source_id_count)
        logger.info("synapse source max count is %d" % (syn_source_max))
        for syn_id, syn in syns_dict.items():
            count = syn_source_count[syn.source.gid]
            syn_source_pctile[syn_id] = float(count) / float(syn_source_max)
        syns_dict = {
            syn_id: syn
            for syn_id, syn in syns_dict.items()
            if syn_source_pctile[syn_id] >= syn_source_threshold
        }
    for syn_id, syn in syns_dict.items():
        syn_sec_dict[syn.syn_section].append(syn)
    syn_xyz_sec_dict = {}
    syn_src_sec_dict = {}
    for sec_id, syns in syn_sec_dict.items():
        sec = biophys_cell.hoc_cell.sections[sec_id]
        syn_locs = [syn.syn_loc for syn in syns]
        ip_x, ip_y, ip_z, ip_diam = interplocs(sec)
        syn_xyz_sec_dict[sec_id] = np.column_stack(
            (ip_x(syn_locs), ip_y(syn_locs), ip_z(syn_locs))
        )
        syn_sources = [syn.source.gid for syn in syns]
        syn_src_sec_dict[sec_id] = np.asarray(syn_sources)

    fig = None
    if plot_method == "mayavi":
        from mayavi import mlab

        colormap = kwargs.get("colormap", "coolwarm")
        mlab.figure(bgcolor=kwargs.get("bgcolor", (0, 0, 0)))

        xcoords = np.asarray(
            [x for (i, x) in morph_graph.nodes.data("x")], dtype=np.float32
        )
        ycoords = np.asarray(
            [y for (i, y) in morph_graph.nodes.data("y")], dtype=np.float32
        )
        zcoords = np.asarray(
            [z for (i, z) in morph_graph.nodes.data("z")], dtype=np.float32
        )
        layer = np.asarray(
            [layer for (i, layer) in morph_graph.nodes.data("layer")],
            dtype=np.int32,
        )

        # edges = nx.minimum_spanning_tree(morph_graph).edges(data=True)
        edges = morph_graph.edges(data=True)
        start_idx, end_idx, _ = np.array(list(edges)).T
        start_idx = start_idx.astype(np.int)
        end_idx = end_idx.astype(np.int)
        # edge_scalars = layers[start_idx]

        logger.info(f"plotting tree {biophys_cell.gid}")

        # Plot morphology graph with Mayavi
        plot_graph(
            xcoords,
            ycoords,
            zcoords,
            start_idx,
            end_idx,
            edge_color=(1, 1, 1),
            opacity=0.6,
            line_width=line_width,
        )

        logger.info(f"plotting {len(syns_dict)} synapses")
        for sec_id, syn_xyz in syn_xyz_sec_dict.items():
            syn_sources = syn_src_sec_dict[sec_id]
            if None in syn_sources:
                mlab.points3d(
                    syn_xyz[:, 0],
                    syn_xyz[:, 1],
                    syn_xyz[:, 2],
                    scale_mode="vector",
                    colormap=colormap,
                    scale_factor=10.0,
                    color=(1, 0, 0),
                )
            else:
                mlab.points3d(
                    syn_xyz[:, 0],
                    syn_xyz[:, 1],
                    syn_xyz[:, 2],
                    syn_sources,
                    scale_mode="vector",
                    colormap=colormap,
                    scale_factor=10.0,
                    color=(1, 0, 0),
                )

        mlab.gcf().scene.x_plus_view()
        mlab.show()

        fig = mlab.gcf()

    elif plot_method == "matplotlib":
        fig = plt.figure(figsize=fig_options.figSize)
        ax = fig.add_subplot(projection="3d")

        xcoords = np.asarray(
            [x for (i, x) in morph_graph.nodes.data("x")], dtype=np.float32
        )
        ycoords = np.asarray(
            [y for (i, y) in morph_graph.nodes.data("y")], dtype=np.float32
        )
        zcoords = np.asarray(
            [z for (i, z) in morph_graph.nodes.data("z")], dtype=np.float32
        )
        layer = np.asarray(
            [layer for (i, layer) in morph_graph.nodes.data("layer")],
            dtype=np.int32,
        )

        sct = ax.scatter(
            xcoords,
            ycoords,
            zcoords,
            c=layer,
            alpha=0.7,
        )
        # produce a legend with the unique colors from the scatter
        legend_elements = sct.legend_elements()
        layer_legend = ax.legend(*legend_elements, loc="upper right", title="Layer")
        ax.add_artist(layer_legend)

        for i, j in morph_graph.edges:
            e_x = (xcoords[i], xcoords[j])
            e_y = (ycoords[i], ycoords[j])
            e_z = (zcoords[i], zcoords[j])

            ax.plot(e_x, e_y, e_z, c="black", alpha=0.5)

        for sec_id, syn_xyz in syn_xyz_sec_dict.items():
            syn_sources = syn_src_sec_dict[sec_id]
            if None in syn_sources:
                ax.scatter(
                    syn_xyz[:, 0],
                    syn_xyz[:, 1],
                    syn_xyz[:, 2],
                    marker="^",
                    s=100,
                )
            else:
                ax.scatter(
                    syn_xyz[:, 0],
                    syn_xyz[:, 1],
                    syn_xyz[:, 2],
                    c=syn_sources,
                    marker="o",
                )

        ax.view_init(30)
        ax.set_axis_off

        # if fig_options.saveFig:
        #    if isinstance(fig_options.saveFig, str):
        #        filename = fig_options.saveFig
        #    else:
        #        filename = (
        #            f"{population}_{gid}_cell_tree.{fig_options.figFormat}"
        #        )
        #    plt.savefig(filename)

        if fig_options.showFig:
            # show_figure()
            plt.show()
    else:
        sl = h.SectionList([sec for sec in biophys_cell.hoc_cell.all])
        for sec in sl:
            sec.v = 0
        h.topology()
        h.psection(list(sl)[0])
        ps = h.PlotShape(sl, False)  # False tells h.PlotShape not to use NEURON's gui
        ax = ps.plot(plt)
        plt.show()

    # return fig


# =============================================================================
# Get radially averaged PSD of 2D PSD (total power spectrum by angular bin)
# =============================================================================
def get_RPSD(psd2D, dTheta=30, rMin=10, rMax=100):
    h = psd2D.shape[0]
    w = psd2D.shape[1]
    wc = w // 2
    hc = h // 2

    # note that displaying PSD as image inverts Y axis
    # create an array of integer angular slices of dTheta
    Y, X = np.ogrid[0:h, 0:w]
    theta = np.rad2deg(np.arctan2(-(Y - hc), (X - wc)))
    theta = np.mod(theta + dTheta / 2 + 360, 360)
    theta = dTheta * (theta // dTheta)
    theta = theta.astype(np.int)

    # mask below rMin and above rMax by setting to -100
    R = np.hypot(-(Y - hc), (X - wc))
    mask = np.logical_and(R > rMin, R < rMax)
    theta = theta + 100
    theta = np.multiply(mask, theta)
    theta = theta - 100

    # SUM all psd2D pixels with label 'theta' for 0<=theta❤60 between rMin and rMax
    angF = np.arange(0, 360, int(dTheta))
    psd1D = ndimage.sum(psd2D, theta, index=angF)

    # normalize each sector to the total sector power
    pwrTotal = np.sum(psd1D)
    psd1D = psd1D / pwrTotal

    return angF, psd1D


def plot_2D_rate_map(
    x,
    y,
    rate_map,
    x0=None,
    y0=None,
    peak_rate=None,
    title=None,
    fft_vmax=10.0,
    density_bin_size=10.0,
    **kwargs,
):
    """

    :param x: array
    :param y: array
    :param rate_map: array
    :param peak_rate: float
    :param title: str
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    if peak_rate is None:
        peak_rate = np.max(rate_map)

    fig = plt.figure(constrained_layout=True, figsize=fig_options.figSize)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2, 1, 1])

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    ax = fig.add_subplot(gs[0, 0])
    pc = ax.pcolor(x, y, rate_map, vmin=0.0, vmax=peak_rate, cmap=fig_options.colormap)
    cbar = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Firing Rate (Hz)",
        rotation=270.0,
        labelpad=20.0,
        fontsize=fig_options.fontSize,
    )
    ax.set_title("Rate Map")
    ax.set_aspect("equal")
    ax.set_xlabel("X Position (cm)", fontsize=fig_options.fontSize)
    ax.set_ylabel("Y Position (cm)", fontsize=fig_options.fontSize)
    ax.tick_params(labelsize=fig_options.fontSize)
    clean_axes(ax)

    if x0 is not None and y0 is not None:
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title("Point Density")
        plot_2D_point_density(np.column_stack((x0, y0)), ax=ax)

    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    psd2D = np.abs(
        np.fft.fftshift(np.fft.fft2(rate_map - np.mean(rate_map)) / rate_map.shape[0])
    )
    im = ax.imshow(
        psd2D,
        vmax=fft_vmax,
        cmap=fig_options.colormap,
        extent=[x_min, x_max, y_min, y_max],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Power", rotation=270.0, labelpad=20.0, fontsize=fig_options.fontSize
    )
    ax.set_title("Rate Periodogram")
    ax.set_aspect("equal")

    angF, rpsd = get_RPSD(psd2D)
    if x0 is not None and y0 is not None:
        ax = fig.add_subplot(gs[1, 1])
    else:
        ax = fig.add_subplot(gs[:, 1])
    sct = ax.scatter(angF, rpsd, cmap=fig_options.colormap)
    ax.set_title("Radially Averaged Spectrogram")

    if title is not None:
        fig.suptitle(title, fontsize=fig_options.fontSize)

    if fig_options.saveFig is not None:
        save_figure(fig_options.saveFig, fig=fig, **fig_options())

    if fig_options.showFig:
        plt.show()

    return fig


def plot_2D_histogram(
    hist,
    x_edges,
    y_edges,
    norm=None,
    ylabel=None,
    xlabel=None,
    title=None,
    cbar_label=None,
    cbar=True,
    vmin=0.0,
    vmax=None,
    **kwargs,
):
    """

    :param hist: ndarray
    :param x_edges: ndarray
    :param y_edges: ndarray
    :param norm: ndarray; optionally normalize hist by nonzero elements of norm array
    :param ylabel: str
    :param xlabel: str
    :param title: str
    :param cbar_label: str
    :param cbar: bool
    :param vmin: float
    :param vmax: float
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    H = np.copy(hist)
    if norm is not None:
        non_zero = np.where(norm > 0.0)
        H[non_zero] = np.divide(H[non_zero], norm[non_zero])

    if vmax is None:
        vmax = np.max(H)
    fig, axes = plt.subplots(figsize=fig_options.figSize)

    pcm_cmap = None
    pcm_boundaries = np.arange(vmin, vmax, 0.1)
    if len(pcm_boundaries) > 0:
        cmap_pls = plt.cm.get_cmap(fig_options.colormap, len(pcm_boundaries))
        pcm_colors = list(cmap_pls(np.arange(len(pcm_boundaries))))
        pcm_cmap = mpl.colors.ListedColormap(pcm_colors[:-1], "")
        pcm_cmap.set_under(pcm_colors[0], alpha=0.0)

    pcm = axes.pcolormesh(x_edges, y_edges, H.T, vmin=vmin, vmax=vmax, cmap=pcm_cmap)

    axes.set_aspect("equal")
    axes.tick_params(labelsize=fig_options.fontSize)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    if cbar:
        cb = fig.colorbar(pcm, cax=cax)
        cb.ax.tick_params(labelsize=fig_options.fontSize)
        if cbar_label is not None:
            cb.set_label(
                cbar_label,
                rotation=270.0,
                labelpad=20.0,
                fontsize=fig_options.fontSize,
            )
    if xlabel is not None:
        axes.set_xlabel(xlabel, fontsize=fig_options.fontSize)
    if ylabel is not None:
        axes.set_ylabel(ylabel, fontsize=fig_options.fontSize)
    if title is not None:
        axes.set_title(title, fontsize=fig_options.fontSize)
    clean_axes(axes)

    if fig_options.saveFig is not None:
        save_figure(fig_options.saveFig, fig=fig, **fig_options())

    if fig_options.showFig:
        plt.show()

    return fig


def plot_2D_point_density(data, width=100, height=100, ax=None, inc=0.3):
    def points_image(data, height, width, inc=0.3):
        xlims = (data[:, 0].min(), data[:, 0].max())
        ylims = (data[:, 1].min(), data[:, 1].max())
        dxl = xlims[1] - xlims[0]
        dyl = ylims[1] - ylims[0]

        img = np.zeros((height + 1, width + 1))
        for i, p in enumerate(data):
            x0 = int(round(((p[0] - xlims[0]) / dxl) * width))
            y0 = int(round((1 - (p[1] - ylims[0]) / dyl) * height))
            img[y0, x0] += inc
            if img[y0, x0] > 1.0:
                img[y0, x0] = 1.0

        return xlims, ylims, img

    if width is None:
        width = int(round(data[:, 0].max() - data[:, 0].min()))
    if height is None:
        height = int(round(data[:, 1].max() - data[:, 1].min()))

    xlims, ylims, img = points_image(data, height=height, width=width, inc=inc)
    ax_extent = list(xlims) + list(ylims)

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(
        img,
        vmin=0,
        vmax=1,
        cmap=plt.get_cmap("hot"),
        interpolation="hermite",
        aspect="auto",
        extent=ax_extent,
    )


## Plot intracellular state trace
def plot_intracellular_state(
    input_path,
    namespace_ids,
    include=["eachPop"],
    time_range=None,
    time_variable="t",
    state_variable="v",
    max_units=1,
    gid_set=None,
    n_trials=1,
    labels="legend",
    lowpass_plot=None,
    reduce=False,
    distance=False,
    **kwargs,
):
    """
    Line plot of intracellular state variable (default: v). Returns the figure handle.

    input_path: file with state data
    namespace_ids: attribute namespaces
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    state_variable: Name of state variable (default: 'v')
    max_units (int): maximum number of units from each population that will be plotted  (default: 1)
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    """

    if reduce and distance:
        raise RuntimeError(
            "plot_intracellular_state: reduce and distance are mutually exclusive"
        )

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    (population_ranges, N) = read_population_ranges(input_path)
    population_names = read_population_names(input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    _, state_info = statedata.query_state(
        input_path, population_names, namespace_ids=namespace_ids
    )

    # Replace 'eachPop' with list of populations
    if "eachPop" in include:
        include.remove("eachPop")
        for pop in population_names:
            include.append(pop)

    if gid_set is None:
        for population in include:
            for namespace in namespace_ids:
                if (population in state_info) and (namespace in state_info[population]):
                    ns_state_info_dict = dict(state_info[population][namespace])
                    if state_variable in ns_state_info_dict:
                        gid_set = list(ns_state_info_dict[state_variable])[:max_units]
                        break
                    else:
                        raise RuntimeError(
                            "unable to find recording for state variable %s population %s namespace %s"
                            % (state_variable, population, namespace)
                        )

    pop_states_dict = defaultdict(lambda: defaultdict(lambda: dict()))
    for namespace_id in namespace_ids:
        logger.info(f"Reading state values from namespace {namespace_id}...")
        data = statedata.read_state(
            input_path,
            include,
            namespace_id,
            time_variable=time_variable,
            state_variables=[state_variable],
            time_range=time_range,
            max_units=max_units,
            gid=gid_set,
            n_trials=n_trials,
        )
        states = data["states"]

        for pop_name, pop_states in states.items():
            for gid, cell_states in pop_states.items():
                pop_states_dict[pop_name][gid][namespace_id] = cell_states

    pop_state_mat_dict = defaultdict(lambda: dict())
    for pop_name, pop_states in pop_states_dict.items():
        for gid, cell_state_dict in pop_states.items():
            nss = sorted(cell_state_dict.keys())
            cell_state_x = cell_state_dict[nss[0]][time_variable]
            cell_state_mat = np.matrix(
                [
                    np.mean(
                        np.row_stack(cell_state_dict[ns][state_variable]),
                        axis=0,
                    )
                    for ns in nss
                ],
                dtype=np.float32,
            )
            cell_state_distances = [cell_state_dict[ns]["distance"] for ns in nss]
            cell_state_ri = [cell_state_dict[ns]["ri"] for ns in nss]
            cell_state_labels = [f"{ns} {state_variable}" for ns in nss]
            pop_state_mat_dict[pop_name][gid] = (
                cell_state_x,
                cell_state_mat,
                cell_state_labels,
                cell_state_distances,
                cell_state_ri,
            )

    stplots = []

    fig, ax, ax_lowpass = None, None, None
    if lowpass_plot is None:
        fig, ax = plt.subplots(figsize=fig_options.figSize, sharex="all", sharey="all")
    elif lowpass_plot == "subplot":
        fig, (ax, ax_lowpass) = plt.subplots(
            nrows=2, figsize=fig_options.figSize, sharex="all", sharey="all"
        )
    else:
        fig, ax = plt.subplots(figsize=fig_options.figSize, sharex="all", sharey="all")
        ax_lowpass = ax

    legend_labels = []
    for pop_name, pop_states in pop_state_mat_dict.items():
        for gid, cell_state_mat in pop_states.items():
            m, n = cell_state_mat[1].shape
            st_x = cell_state_mat[0][0].reshape((n,))

            if distance:
                cell_state_distances = cell_state_mat[3]
                logger.info(f"cell_state_distances = {cell_state_distances}")
                cell_state_ri = cell_state_mat[4]
                distance_rank = np.argsort(cell_state_distances, kind="stable")
                distance_rank_descending = distance_rank[::-1]
                state_rows = []
                for i in range(0, m):
                    j = distance_rank_descending[i]
                    state_rows.append(np.asarray(cell_state_mat[1][j, :]).reshape((n,)))
                state_mat = np.row_stack(state_rows)
                d = np.asarray(cell_state_distances)[distance_rank_descending]
                ri = np.asarray(cell_state_ri)[distance_rank_descending]
                pcm = ax.pcolormesh(st_x, d, state_mat, cmap=fig_options.colormap)
                cb = fig.colorbar(pcm, ax=ax, shrink=0.9, aspect=20)
                stplots.append(pcm)
                legend_labels.append(f"{pop_name} {gid}")

            else:
                cell_states = [
                    np.asarray(cell_state_mat[1][i, :]).reshape((n,)) for i in range(m)
                ]

                if reduce:
                    cell_state = np.mean(np.vstack(cell_states), axis=0)
                    (line,) = ax.plot(st_x, cell_state)
                    stplots.append(line)
                    logger.info(
                        f"plot_state: min/max/mean value is "
                        f"{np.min(cell_state):.02f} / {np.max(cell_state):.02f} / "
                        f"{np.mean(cell_state):.02f}"
                    )
                else:
                    for i, cell_state in enumerate(cell_states):
                        (line,) = ax.plot(st_x, cell_state)
                        stplots.append(line)
                        logger.info(
                            f"plot_state: min/max/mean value of state {i} is "
                            f"{np.min(cell_state):.02f} / {np.max(cell_state):.02f} "
                            f"/ {np.mean(cell_state):.02f}"
                        )

                        if cell_state_mat[3][i] is not None:
                            legend_labels.append(
                                f"{pop_name} {gid} "
                                f"{cell_state_mat[2][i]} ({cell_state_mat[3][i]:.02f} um)"
                            )
                        else:
                            legend_labels.append(
                                f"{pop_name} {gid} " f"{cell_state_mat[2][i]}"
                            )

                if lowpass_plot is not None and not distance:
                    try:
                        filtered_cell_states = [
                            get_low_pass_filtered_trace(cell_state, st_x)
                            for cell_state in cell_states
                        ]
                        mean_filtered_cell_state = np.mean(filtered_cell_states, axis=0)
                        ax_lowpass.plot(
                            st_x,
                            mean_filtered_cell_state,
                            label=f"{pop_name} {gid} (filtered)",
                            linewidth=fig_options.lw,
                            alpha=0.75,
                        )
                    except:
                        pass

            ax.set_xlabel("Time [ms]", fontsize=fig_options.fontSize)
            if distance:
                ax.set_ylabel("distance from soma [um]", fontsize=fig_options.fontSize)
            else:
                ax.set_ylabel(state_variable, fontsize=fig_options.fontSize)
            # ax.legend()

    # Add legend
    if labels == "legend":
        lgd = plt.legend(
            stplots,
            legend_labels,
            fontsize=fig_options.fontSize,
            scatterpoints=1,
            markerscale=5.0,
            loc="upper right",
            bbox_to_anchor=(0.5, 1.0),
        )
        ## From https://stackoverflow.com/questions/30413789/matplotlib-automatic-legend-outside-plot
        ## draw the legend on the canvas to assign it real pixel coordinates:
        plt.gcf().canvas.draw()
        ## transformation from pixel coordinates to Figure coordinates:
        transfig = plt.gcf().transFigure.inverted()
        ## Get the legend extents in pixels and convert to Figure coordinates.
        ## Pull out the farthest extent in the x direction since that is the canvas direction we need to adjust:
        lgd_pos = lgd.get_window_extent()
        lgd_coord = transfig.transform(lgd_pos)
        lgd_xmax = lgd_coord[1, 0]
        ## Do the same for the Axes:
        ax_pos = plt.gca().get_window_extent()
        ax_coord = transfig.transform(ax_pos)
        ax_xmax = ax_coord[1, 0]
        ## Adjust the Figure canvas using tight_layout for
        ## Axes that must move over to allow room for the legend to fit within the canvas:
        shift = 1 - (lgd_xmax - ax_xmax)
        plt.gcf().tight_layout(rect=(0, 0, shift, 1))

    # save figure
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = input_path + " " + f"state.{fig_options.figFormat}"
            plt.savefig(filename)

    # show fig
    if fig_options.showFig:
        show_figure()

    return fig


def plot_network_clamp(
    input_path,
    spike_namespace,
    intracellular_namespace,
    gid,
    target_input_features_path=None,
    target_input_features_namespace=None,
    target_input_features_arena_id=None,
    target_input_features_trajectory_id=None,
    config_file=None,
    config_prefix="",
    include="eachPop",
    include_artificial=True,
    time_range=None,
    time_variable="t",
    intracellular_variable="v",
    labels="overlay",
    pop_rates=True,
    all_spike_hist=False,
    spike_hist_bin=5,
    lowpass_plot_type="overlay",
    n_trials=-1,
    marker=".",
    opt_seed=None,
    **kwargs,
):
    """
    Raster plot of target cell intracellular trace + spike raster of presynaptic inputs. Returns the figure handle.

    input_path: file with spike data
    spike_namespace: attribute namespace for spike events
    intracellular_namespace: attribute namespace for intracellular trace
    target_input_features_path: optional file with input features
    target_input_features_namespaces: optional attribute namespace for input features
    config_path: path to network configuration file; required when target_input_features_path is given
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    pop_rates = (True|False): Include population rates (default: False)
    spike_hist_bin (int): Size of bin in ms to use for histogram (default: 5)
    marker (char): Marker for each spike (default: '.')
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    (population_ranges, N) = read_population_ranges(input_path)
    population_names = read_population_names(input_path)

    _, state_info = statedata.query_state(
        input_path, population_names, namespace_ids=[intracellular_namespace]
    )

    state_pop_name = None
    pop_num_cells = {}
    pop_start_inds = {}

    for population in population_names:
        pop_start_inds[population] = population_ranges[population][0]
        pop_range = population_ranges[population]
        pop_num_cells[population] = pop_range[1]

    if gid is None:
        for population in state_info.keys():
            if intracellular_namespace in state_info[population]:
                state_pop_name = population
                gid = dict(state_info[population][intracellular_namespace])[
                    intracellular_variable
                ][0]
                break
    else:
        for population in population_names:
            pop_range = population_ranges[population]
            if (gid >= pop_range[0]) and (gid < pop_range[0] + pop_range[1]):
                state_pop_name = population
                break

    # Replace 'eachPop' with list of populations
    if "eachPop" in include:
        include.remove("eachPop")
        for pop in population_names:
            include.append(pop)

    spk_include = include
    if (state_pop_name is not None) and (state_pop_name not in spk_include):
        spk_include.append(state_pop_name)

    # sort according to start index
    include.sort(key=lambda x: pop_start_inds[x])
    include.reverse()

    sys.stdout.flush()
    spkdata = spikedata.read_spike_events(
        input_path,
        spk_include,
        spike_namespace,
        spike_train_attr_name=time_variable,
        time_range=time_range,
        n_trials=n_trials,
        include_artificial=include_artificial,
    )
    logger.info(
        "plot_network_clamp: reading recorded intracellular variable %s for gid %d"
        % (intracellular_variable, gid)
    )
    indata = statedata.read_state(
        input_path,
        [state_pop_name],
        intracellular_namespace,
        time_variable=time_variable,
        state_variables=[intracellular_variable],
        time_range=time_range,
        gid=[gid],
        n_trials=n_trials,
    )

    spkpoplst = spkdata["spkpoplst"]
    spkindlst = spkdata["spkindlst"]
    spktlst = spkdata["spktlst"]
    num_cell_spks = spkdata["num_cell_spks"]
    pop_active_cells = spkdata["pop_active_cells"]
    tmin = spkdata["tmin"]
    tmax = spkdata["tmax"]
    n_trials = spkdata["n_trials"]

    if time_range is None:
        time_range = [tmin, tmax]

    if (
        time_range[0] == time_range[1]
        or time_range[0] == float("inf")
        or time_range[1] == float("inf")
    ):
        raise RuntimeError(f"plot_network_clamp: invalid time_range: {time_range}")
    time_bins = np.arange(time_range[0], time_range[1], spike_hist_bin)

    baks_config = copy.copy(kwargs)
    target_rate = None
    target_rate_time = None
    target_rate_ip = None
    if (target_input_features_path is not None) and (
        target_input_features_namespace is not None
    ):
        if config_file is None:
            raise RuntimeError(
                "plot_network_clamp: config_file must be provided with target_input_features_path."
            )
        env = Env(
            config_file=config_file,
            arena_id=target_input_features_arena_id,
            trajectory_id=target_input_features_trajectory_id,
            config_prefix=config_prefix,
        )

        if env.analysis_config is not None:
            baks_config.update(env.analysis_config["Firing Rate Inference"])

        target_trj_rate_maps = stimulus.rate_maps_from_features(
            env,
            state_pop_name,
            cell_index_set=[gid],
            input_features_path=target_input_features_path,
            input_features_namespace=target_input_features_namespace,
            time_range=time_range,
            include_time=True,
        )
        target_rate_time, target_rate = target_trj_rate_maps[gid]
        target_rate_ip = interpolate.Akima1DInterpolator(target_rate_time, target_rate)

    maxN = 0
    minN = N

    avg_rates = {}
    tsecs = (time_range[1] - time_range[0]) / 1e3
    for i, pop_name in enumerate(spkpoplst):
        pop_num = len(pop_active_cells[pop_name])
        maxN = max(maxN, max(pop_active_cells[pop_name]))
        minN = min(minN, min(pop_active_cells[pop_name]))
        if pop_num > 0:
            if num_cell_spks[pop_name] == 0:
                avg_rates[pop_name] = 0
            else:
                avg_rates[pop_name] = (
                    num_cell_spks[pop_name] / pop_num / n_trials
                ) / tsecs

    pop_colors = {
        pop_name: dflt_colors[ipop % len(dflt_colors)]
        for ipop, pop_name in enumerate(spkpoplst)
    }

    pop_spk_dict = {
        pop_name: (pop_spkinds, pop_spkts)
        for (pop_name, pop_spkinds, pop_spkts) in zip(spkpoplst, spkindlst, spktlst)
    }
    N = pop_num_cells[pop_name]
    S = pop_start_inds[pop_name]

    n_plots = len(spkpoplst) + 2
    plot_height_ratios = [1] * len(spkpoplst)
    if all_spike_hist:
        n_plots += 1
        plot_height_ratios.append(1)

    # Target spike plot
    plot_height_ratios.append(1)

    if target_rate_ip is not None:
        n_plots += 2
        plot_height_ratios.append(0.5)
        plot_height_ratios.append(0.5)

    # State plot
    plot_height_ratios.append(2)

    if lowpass_plot_type == "subplot":
        n_plots += 1
        plot_height_ratios.append(1)

    fig, axes = plt.subplots(
        nrows=n_plots,
        sharex=True,
        figsize=fig_options.figSize,
        gridspec_kw={"height_ratios": plot_height_ratios},
    )

    stplots = []

    def sphist(trial_spkts):
        if len(trial_spkts) > 0:
            bin_edges = np.histogram_bin_edges(
                trial_spkts[0],
                bins=np.arange(time_range[0], time_range[1], spike_hist_bin),
            )
            trial_sphist_ys = np.array(
                [np.histogram(spkts, bins=bin_edges)[0] for spkts in trial_spkts]
            )
            sphist_y = np.mean(trial_sphist_ys, axis=0)

            sphist_x = bin_edges[:-1] + (spike_hist_bin / 2)

            pch = interpolate.pchip(sphist_x, sphist_y)
            res_npts = int(sphist_x.max() - sphist_x.min())
            sphist_x_res = np.linspace(
                sphist_x.min(), sphist_x.max(), res_npts, endpoint=True
            )
            sphist_y_res = pch(sphist_x_res)
        else:
            bin_edges = np.arange(time_range[0], time_range[1], spike_hist_bin)
            sphist_x_res = bin_edges[:-1] + (spike_hist_bin / 2)
            sphist_y_res = np.zeros(sphist_x_res.shape)

        return sphist_x_res, sphist_y_res

    for i, pop_name in enumerate(include):
        pop_spkinds, pop_spkts = pop_spk_dict.get(pop_name, ([], []))

        sphist_x, sphist_y = sphist(pop_spkts)
        sph = axes[i].fill_between(
            sphist_x,
            sphist_y,
            linewidth=fig_options.lw,
            color=pop_colors.get(pop_name, dflt_colors[0]),
            alpha=0.5,
            label=pop_name,
        )
        axes[i].set_ylim(0.0, np.ceil(np.max(sphist_y)))
        stplots.append(sph)

        if i == 0:
            axes[i].set_xlim(time_range)
            axes[i].set_xlabel("Time (ms)", fontsize=fig_options.fontSize)
            axes[i].set_ylabel("Spike Count", fontsize=fig_options.fontSize)

    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-2]], visible=False)

    # set raster plot ticks to the end of the index range for each population
    for i, pop_name in enumerate(include):
        a = fig.axes[i]
        start, end = a.get_ylim()
        a.get_yaxis().set_ticks([end])

    # set raster plot ticks to start and end of index range for first population
    a = fig.axes[len(spkpoplst) - 1]
    start, end = a.get_ylim()
    a.get_yaxis().set_ticks([start, end])

    if pop_rates:
        lgd_labels = [
            pop_name
            + " (%i active; %.3g Hz)"
            % (len(pop_active_cells[pop_name]), avg_rates[pop_name])
            for pop_name in spkpoplst
            if pop_name in avg_rates
        ]
    else:
        lgd_labels = [
            pop_name + f" ({len(pop_active_cells[pop_name])} active)"
            for pop_name in spkpoplst
            if pop_name in avg_rates
        ]

    i_ax = len(spkpoplst)

    if spktlst:
        if all_spike_hist:
            # Calculate and plot total spike histogram
            all_trial_spkts = [list() for i in range(len(spktlst[0]))]
            for i, pop_name in enumerate(include):
                pop_spkinds, pop_spkts = pop_spk_dict.get(pop_name, ([], []))
                for trial_i, this_trial_spkts in enumerate(pop_spkts):
                    all_trial_spkts[trial_i].append(this_trial_spkts)
            merged_trial_spkts = [
                np.concatenate(trial_spkts, axis=0) for trial_spkts in all_trial_spkts
            ]
            sphist_x, sphist_y = sphist(merged_trial_spkts)
            sprate = np.sum(avg_rates[pop_name] for pop_name in avg_rates) / len(
                avg_rates
            )
            ax_spk = axes[i_ax]
            ax_spk.plot(sphist_x, sphist_y, linewidth=1.0)
            ax_spk.set_xlabel("Time (ms)", fontsize=fig_options.fontSize)
            ax_spk.set_xlim(time_range)
            ax_spk.set_ylim((np.min(sphist_y), np.max(sphist_y) * 2))
            if pop_rates:
                lgd_label = f"mean firing rate: {sprate:.3g} Hz"
                at = AnchoredText(
                    lgd_label,
                    loc="upper right",
                    borderpad=0.01,
                    prop=dict(size=fig_options.fontSizej),
                )
                ax_spk.add_artist(at)
            i_ax += 1

        # Calculate and plot spike histogram for target gid
        pop_spkinds, pop_spkts = pop_spk_dict.get(state_pop_name, ([], []))
        trial_sdf_ips = []
        spk_count = 0
        ax_spk = axes[i_ax]
        for this_trial_spkinds, this_trial_spkts in zip_longest(pop_spkinds, pop_spkts):
            spk_inds = np.argwhere(this_trial_spkinds == gid)
            spk_count += len(spk_inds)
            if target_rate_ip is not None:
                sdf_dict = spikedata.spike_density_estimate(
                    state_pop_name,
                    {gid: this_trial_spkts[spk_inds]},
                    time_bins,
                    **baks_config,
                )
                trial_sdf_rate = sdf_dict[gid]["rate"]
                trial_sdf_time = sdf_dict[gid]["time"]
                trial_sdf_ip = interpolate.Akima1DInterpolator(
                    trial_sdf_time, trial_sdf_rate
                )
                trial_sdf_ips.append(trial_sdf_ip)
            if len(spk_inds) > 0:
                ax_spk.stem(
                    this_trial_spkts[spk_inds],
                    [0.5] * len(spk_inds),
                    markerfmt=" ",
                )
            ax_spk.set_yticks([])
        sprate = spk_count / n_trials / tsecs

        ax_spk.set_xlabel("Time (ms)", fontsize=fig_options.fontSize)
        ax_spk.set_xlim(time_range)
        if pop_rates:
            lgd_label = "%s gid %d: %.3g Hz" % (state_pop_name, gid, sprate)
            at = AnchoredText(
                lgd_label,
                loc="upper right",
                borderpad=0.01,
                prop=dict(size=fig_options.fontSize),
            )
            ax_spk.add_artist(at)
        i_ax += 1

    if target_rate is not None:
        t = np.arange(time_range[0], time_range[1], 1.0)
        target_rate_t_range = target_rate_ip(t)
        if np.any(np.isnan(target_rate_t_range)):
            target_rate_t_range[np.isnan(target_rate_t_range)] = 0.0
        vmin, vmax = 0, np.max(target_rate_t_range)
        ax_target_rate = axes[i_ax]
        i_ax += 1
        ax_target_rate.imshow(
            target_rate_t_range[np.newaxis, :],
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax_target_rate.set_yticks([])
        ax_mean_sdf = axes[i_ax]
        i_ax += 1
        if len(trial_sdf_ips) > 0:
            trial_sdf_matrix = np.row_stack(
                [trial_sdf_ip(t) for trial_sdf_ip in trial_sdf_ips]
            )
            mean_sdf = np.mean(trial_sdf_matrix, axis=0)
            ax_mean_sdf.imshow(
                mean_sdf[np.newaxis, :], vmin=vmin, vmax=vmax, aspect="auto"
            )
        ax_mean_sdf.set_yticks([])

    # Plot intracellular state
    ax_state = axes[i_ax]
    ax_state.set_xlabel("Time (ms)", fontsize=fig_options.fontSize)
    ax_state.set_ylabel(intracellular_variable, fontsize=fig_options.fontSize)
    ax_state.set_xlim(time_range)
    i_ax += 1

    # Plot lowpass-filtered intracellular state if lowpass_plot_type is set to subplot
    if lowpass_plot_type == "subplot":
        ax_lowpass = axes[i_ax]
        i_ax += 1
    else:
        ax_lowpass = ax_state

    states = indata["states"]
    stvplots = []

    for pop_name, pop_states in states.items():
        for gid, cell_states in pop_states.items():
            st_len = cell_states[intracellular_variable][0].shape[0]
            st_xs = [x[:st_len] for x in cell_states[time_variable]]
            st_ys = [y[:st_len] for y in cell_states[intracellular_variable]]
            st_x = st_xs[0]
            try:
                filtered_st_ys = [
                    get_low_pass_filtered_trace(st_y, st_x)
                    for st_x, st_y in zip(st_xs, st_ys)
                ]
                filtered_st_y = np.mean(filtered_st_ys, axis=0)
                ax_lowpass.plot(
                    st_x,
                    filtered_st_y,
                    label=f"{pop_name} (filtered)",
                    linewidth=fig_options.lw,
                    alpha=0.75,
                )
            except:
                pass

            for st_y in st_ys:
                stvplots.append(
                    ax_state.plot(
                        st_x,
                        st_y,
                        label=pop_name,
                        linewidth=fig_options.lw,
                        alpha=0.5,
                    )
                )

    if labels == "legend":
        # Shrink axes by 15%
        for ax in axes:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # Add legend
        lgd = fig.legend(
            stplots,
            lgd_labels,
            loc="center right",
            fontsize="small",
            scatterpoints=1,
            markerscale=5.0,
            bbox_to_anchor=(1.002, 0.5),
            bbox_transform=plt.gcf().transFigure,
        )
        fig.artists.append(lgd)

    elif labels == "overlay":
        for i, (pop_name, lgd_label) in enumerate(zip(spkpoplst, lgd_labels)):
            at = AnchoredText(
                lgd_label,
                loc="upper right",
                borderpad=0.01,
                prop=dict(size=fig_options.fontSize),
            )
            axes[i].add_artist(at)
        max_label_len = max(len(l) for l in lgd_labels)

    else:
        raise RuntimeError(f"plot_network_clamp: unknown label type {labels}")

    # save figure
    ts = time.strftime("%Y%m%d_%H%M%S")
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = (
                "Network Clamp %s %i.%s" % (state_pop_name, gid, fig_options.figFormat)
                if opt_seed is None
                else "NetworkClamp_{!s}_{:d}_{!s}_{:08d}.{!s}".format(
                    state_pop_name, gid, ts, opt_seed, fig_options.figFormat
                )
            )
            plt.savefig(filename)

    # show fig
    if fig_options.showFig:
        show_figure()

    return fig


def plot_single_vertex_dist(
    env,
    connectivity_path,
    coords_path,
    distances_namespace,
    target_gid,
    destination,
    source,
    extent_type="local",
    direction="in",
    bin_size=20.0,
    normed=False,
    comm=None,
    **kwargs,
):
    """
    Plot vertex distribution with respect to arc distance for a single postsynaptic cell.

    :param env:
    :param connectivity_path:
    :param coords_path:
    :param distances_namespace:
    :param target_gid:
    :param destination:
    :param source:

    """

    from miv_simulator.geometry.geometry import measure_distance_extents

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    source_soma_distances = bcast_cell_attributes(
        coords_path, source, namespace=distances_namespace, comm=comm, root=0
    )
    destination_soma_distances = bcast_cell_attributes(
        coords_path,
        destination,
        namespace=distances_namespace,
        comm=comm,
        root=0,
    )

    (
        (total_x_min, total_x_max),
        (total_y_min, total_y_max),
    ) = measure_distance_extents(env.geometry, volume=network_volume)

    source_soma_distance_U = {}
    source_soma_distance_V = {}
    destination_soma_distance_U = {}
    destination_soma_distance_V = {}
    for k, v in source_soma_distances:
        source_soma_distance_U[k] = v["U Distance"][0]
        source_soma_distance_V[k] = v["V Distance"][0]
    for k, v in destination_soma_distances:
        destination_soma_distance_U[k] = v["U Distance"][0]
        destination_soma_distance_V[k] = v["V Distance"][0]

    del source_soma_distances
    del destination_soma_distances

    g = NeuroH5ProjectionGen(
        connectivity_path, source, destination, comm=comm, cache_size=20
    )

    dist_bins = {}

    if direction == "in":
        for destination_gid, rest in g:
            if destination_gid == target_gid:
                (source_indexes, attr_dict) = rest
                for source_gid in source_indexes:
                    dist_u = source_soma_distance_U[source_gid]
                    dist_v = source_soma_distance_V[source_gid]
                    update_bins(dist_bins, bin_size, dist_u, dist_v)
                break
    elif direction == "out":
        for destination_gid, rest in g:
            if rest is not None:
                (source_indexes, attr_dict) = rest
                for source_gid in source_indexes:
                    if source_gid == target_gid:
                        dist_u = destination_soma_distance_U[destination_gid]
                        dist_v = destination_soma_distance_V[destination_gid]
                        update_bins(dist_bins, bin_size, dist_u, dist_v)
    else:
        raise RuntimeError(f"Unknown direction type {direction}")

    add_bins_op = MPI.Op.Create(add_bins, commute=True)
    dist_bins = comm.reduce(dist_bins, op=add_bins_op)

    if rank == 0:
        dist_hist_vals, dist_u_bin_edges, dist_v_bin_edges = finalize_bins(
            dist_bins, bin_size
        )

        dist_x_min = dist_u_bin_edges[0]
        dist_x_max = dist_u_bin_edges[-1]
        dist_y_min = dist_v_bin_edges[0]
        dist_y_max = dist_v_bin_edges[-1]

        if extent_type == "local":
            x_min = dist_x_min
            x_max = dist_x_max
            y_min = dist_y_min
            y_max = dist_y_max
        elif extent_type == "global":
            x_min = total_x_min
            x_max = total_x_max
            y_min = total_y_min
            y_max = total_y_max
        else:
            raise RuntimeError(f"Unknown extent type {extent_type}")

        X, Y = np.meshgrid(dist_u_bin_edges, dist_v_bin_edges)

        fig = plt.figure(figsize=fig_options.figSize)

        ax = plt.gca()
        ax.axis([x_min, x_max, y_min, y_max])

        if direction == "in":
            ax.plot(
                destination_soma_distance_U[target_gid],
                destination_soma_distance_V[target_gid],
                "r+",
                markersize=12,
                mew=3,
            )
        elif direction == "out":
            ax.plot(
                source_soma_distance_U[target_gid],
                source_soma_distance_V[target_gid],
                "r+",
                markersize=12,
                mew=3,
            )
        else:
            raise RuntimeError(f"Unknown direction type {direction}")

        H = np.array(dist_hist_vals.todense())
        if normed:
            H = np.divide(H.astype(float), float(np.max(H)))
        pcm_boundaries = np.arange(0, np.max(H), 0.1)
        cmap_pls = plt.cm.get_cmap(fig_options.colormap, len(pcm_boundaries))
        pcm_colors = list(cmap_pls(np.arange(len(pcm_boundaries))))
        pcm_cmap = mpl.colors.ListedColormap(pcm_colors[:-1], "")
        pcm_cmap.set_under(pcm_colors[0], alpha=0.0)
        pcm = ax.pcolormesh(X, Y, H.T, cmap=pcm_cmap)

        clb_label = (
            "Normalized number of connections" if normed else "Number of connections"
        )
        clb = fig.colorbar(pcm, ax=ax, shrink=0.5, label=clb_label)
        clb.ax.tick_params(labelsize=fig_options.fontSize)

        ax.set_aspect("equal")
        ax.set_facecolor(pcm_colors[0])
        ax.tick_params(labelsize=fig_options.fontSize)
        ax.set_xlabel("Longitudinal position (um)", fontsize=fig_options.fontSize)
        ax.set_ylabel("Transverse position (um)", fontsize=fig_options.fontSize)
        ax.set_title(
            f"Connectivity distribution ({direction}) of "
            f"{source} to {destination} for gid {target_gid}",
            fontsize=fig_options.fontSize,
        )

        if fig_options.showFig:
            show_figure()

        plt.tight_layout()
        if fig_options.saveFig:
            if isinstance(fig_options.saveFig, str):
                filename = fig_options.saveFig
            else:
                filename = f"Connection distance {direction} {source} to {destination} gid {target_gid}.{fig_options.figFormat}"
                plt.savefig(filename)

    comm.barrier()


def update_spatial_rasters(
    frame,
    scts,
    timebins,
    n_trials,
    data,
    distances_U_dict,
    distances_V_dict,
    lgd,
):
    N = len(timebins)
    if frame > 0:
        t0 = timebins[frame % N]
        t1 = timebins[(frame + 1) % N]
        trial = frame // N
        for p, (pop_name, spkinds, spkts) in enumerate(data):
            distances_U = distances_U_dict[pop_name]
            distances_V = distances_V_dict[pop_name]
            rinds = np.where(np.logical_and(spkts[trial] >= t0, spkts[trial] <= t1))
            cinds = spkinds[trial][rinds]
            x = np.asarray([distances_U[ind] for ind in cinds])
            y = np.asarray([distances_V[ind] for ind in cinds])
            scts[p].set_data(x, y)
            scts[p].set_label(pop_name)
            if n_trials > 1:
                scts[-1].set_text(f"trial {trial}; t = {t1:.02f} ms")
            else:
                scts[-1].set_text(f"t = {t1:.02f} ms")
    return scts


def init_spatial_rasters(
    ax,
    timebins,
    n_trials,
    data,
    range_U_dict,
    range_V_dict,
    distances_U_dict,
    distances_V_dict,
    lgd,
    marker,
    pop_colors,
    **kwargs,
):
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    scts = []
    t0 = timebins[0]
    t1 = timebins[1]
    min_U = None
    min_V = None
    max_U = None
    max_V = None
    for pop_name, spkinds, spkts in data:
        distances_U = distances_U_dict[pop_name]
        distances_V = distances_V_dict[pop_name]
        rinds = np.where(np.logical_and(spkts[0] >= t0, spkts[0] <= t1))
        cinds = spkinds[0][rinds]
        x = np.asarray([distances_U[ind] for ind in cinds])
        y = np.asarray([distances_V[ind] for ind in cinds])
        # scts.append(ax.scatter(x, y, linewidths=options.lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name))
        scts = scts + plt.plot([], [], marker, animated=True, alpha=0.5)
        if min_U is None:
            min_U = range_U_dict[pop_name][0]
        else:
            min_U = min(min_U, range_U_dict[pop_name][0])
        if min_V is None:
            min_V = range_V_dict[pop_name][0]
        else:
            min_V = min(min_V, range_V_dict[pop_name][0])
        if max_U is None:
            max_U = range_U_dict[pop_name][1]
        else:
            max_U = max(max_U, range_U_dict[pop_name][1])
        if max_V is None:
            max_V = range_V_dict[pop_name][1]
        else:
            max_V = max(max_V, range_V_dict[pop_name][1])
    ax.set_xlim((min_U, max_U))
    ax.set_ylim((min_V, max_V))

    return scts + [
        lgd(scts),
        plt.text(
            0.05,
            0.95,
            "t = %f ms" % t0,
            fontsize=fig_options.fontSize,
            transform=ax.transAxes,
        ),
    ]


spatial_raster_aniplots = []


## Plot spike raster
def plot_spatial_spike_raster(
    input_path,
    namespace_id,
    coords_path,
    distances_namespace="Arc Distances",
    include=["eachPop"],
    time_step=5.0,
    time_range=None,
    time_variable="t",
    include_artificial=True,
    max_spikes=int(1e6),
    marker="o",
    **kwargs,
):
    """
    Spatial raster plot of network spike times. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    max_spikes (int): maximum number of spikes that will be plotted  (default: 1e6)
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    marker (char): Marker for each spike (default: '|')
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    (population_ranges, N) = read_population_ranges(input_path)
    population_names = read_population_names(input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    # Replace 'eachPop' with list of populations
    if "eachPop" in include:
        include.remove("eachPop")
        for pop in population_names:
            include.append(pop)

    distance_U_dict = {}
    distance_V_dict = {}
    range_U_dict = {}
    range_V_dict = {}
    for population in include:
        distances = read_cell_attributes(
            coords_path, population, namespace=distances_namespace
        )

        soma_distances = {
            k: (v["U Distance"][0], v["V Distance"][0]) for (k, v) in distances
        }
        del distances

        logger.info("read distances (%i elements)" % len(soma_distances.keys()))
        distance_U_array = np.asarray(
            [soma_distances[gid][0] for gid in soma_distances]
        )
        distance_V_array = np.asarray(
            [soma_distances[gid][1] for gid in soma_distances]
        )

        U_min = np.min(distance_U_array)
        U_max = np.max(distance_U_array)
        V_min = np.min(distance_V_array)
        V_max = np.max(distance_V_array)

        range_U_dict[population] = (U_min, U_max)
        range_V_dict[population] = (V_min, V_max)

        distance_U = {gid: soma_distances[gid][0] for gid in soma_distances}
        distance_V = {gid: soma_distances[gid][1] for gid in soma_distances}

        distance_U_dict[population] = distance_U
        distance_V_dict[population] = distance_V

    spkdata = spikedata.read_spike_events(
        input_path,
        include,
        namespace_id,
        spike_train_attr_name=time_variable,
        time_range=time_range,
        include_artificial=include_artificial,
    )

    n_trials = spkdata["n_trials"]
    spkpoplst = spkdata["spkpoplst"]
    spkindlst = spkdata["spkindlst"]
    spktlst = spkdata["spktlst"]
    num_cell_spks = spkdata["num_cell_spks"]
    pop_active_cells = spkdata["pop_active_cells"]
    tmin = spkdata["tmin"]
    tmax = spkdata["tmax"]

    time_range = [tmin, tmax]

    pop_colors = {
        pop_name: dflt_colors[ipop % len(dflt_colors)]
        for ipop, pop_name in enumerate(spkpoplst)
    }

    # Plot spikes
    fig, ax = plt.subplots(figsize=fig_options.figSize)

    pop_labels = [pop_name for pop_name in spkpoplst]
    legend_labels = pop_labels
    lgd = lambda objs: plt.legend(
        objs,
        legend_labels,
        fontsize=fig_options.fontSize,
        scatterpoints=1,
        markerscale=2.0,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),
    )

    timebins = np.linspace(tmin, tmax, int(((tmax - tmin) / time_step)))

    data = list(zip(spkpoplst, spkindlst, spktlst))
    scts = init_spatial_rasters(
        ax,
        timebins,
        n_trials,
        data,
        range_U_dict,
        range_V_dict,
        distance_U_dict,
        distance_V_dict,
        lgd,
        marker,
        pop_colors,
    )
    ani = FuncAnimation(
        fig,
        func=update_spatial_rasters,
        frames=list(range(0, len(timebins) * n_trials - 1)),
        blit=True,
        repeat=False,
        init_func=lambda: scts,
        fargs=(
            scts,
            timebins,
            n_trials,
            data,
            distance_U_dict,
            distance_V_dict,
            lgd,
        ),
    )
    spatial_raster_aniplots.append(ani)

    # show fig
    if fig_options.showFig:
        show_figure()

    if fig_options.saveFig:
        Writer = writers["ffmpeg"]
        writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(f"{namespace_id} spatial raster.mp4", writer=writer)

    return fig
