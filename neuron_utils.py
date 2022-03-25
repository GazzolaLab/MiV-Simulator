
import os, os.path, math
import numpy as np
try:
    from mpi4py import MPI  # Must come before importing NEURON
except Exception:
    pass
from dentate.utils import get_module_logger
from neuron import h
from scipy import interpolate

# This logger will inherit its settings from the root logger, created in biophys_microcircuit.env
logger = get_module_logger(__name__)


freq = 100  # Hz, frequency at which AC length constant will be computed
d_lambda = 0.1  # no segment will be longer than this fraction of the AC length constant
default_ordered_sec_types = ['soma', 'hillock', 'ais', 'axon', 'basal', 'trunk', 'apical', 'tuft', 'spine']
default_hoc_sec_lists = {'soma': 'somaidx', 'hillock': 'hilidx', 'ais': 'aisidx', 'axon': 'axonidx',
                         'basal': 'basalidx', 'apical': 'apicalidx', 'trunk': 'trunkidx', 'tuft': 'tuftidx'}

def lambda_f(sec, f=freq):
    """
    Calculates the AC length constant for the given section at the frequency f
    Used to determine the number of segments per hoc section to achieve the desired spatial and temporal resolution
    :param sec: :class:'h.Section'
    :param f: int
    :return: int
    """
    diam = np.mean([seg.diam for seg in sec])
    Ra = sec.Ra
    cm = np.mean([seg.cm for seg in sec])
    return 1e5 * math.sqrt(diam / (4. * math.pi * f * Ra * cm))


def d_lambda_nseg(sec, lam=d_lambda, f=freq):
    """
    The AC length constant for this section and the user-defined fraction is used to determine the maximum size of each
    segment to achieve the desired spatial and temporal resolution. This method returns the number of segments to set
    the nseg parameter for this section. For tapered cylindrical sections, the diam parameter will need to be
    reinitialized after nseg changes.
    :param sec : :class:'h.Section'
    :param lam : int
    :param f : int
    :return : int
    """
    L = sec.L
    return int(((L / (lam * lambda_f(sec, f))) + 0.9) / 2) * 2 + 1


def reinit_diam(sec, diam_bounds):
    """
    For a node associated with a hoc section that is a tapered cylinder, every time the spatial resolution
    of the section (nseg) is changed, the section diameters must be reinitialized. This method checks the
    node's content dictionary for diameter boundaries and recalibrates the hoc section associated with this node.
    """
    if diam_bounds is not None:
        diam1, diam2 = diam_bounds
        h(f'diam(0:1)={diam1}:{diam2}', sec=sec)
        

def init_nseg(sec, spatial_res=0, verbose=True):
    """
    Initializes the number of segments in this section (nseg) based on the AC length constant. Must be re-initialized
    whenever basic cable properties Ra or cm are changed. The spatial resolution parameter increases the number of
    segments per section by a factor of an exponent of 3.
    :param sec: :class:'h.Section'
    :param spatial_res: int
    :param verbose: bool
    """
    sugg_nseg = d_lambda_nseg(sec)
    sugg_nseg *= 3 ** spatial_res
    if verbose:
        logger.info(f'init_nseg: changed {sec.hname()}.nseg {sec.nseg} --> {sugg_nseg}')
    sec.nseg = int(sugg_nseg)


def load_cell_template(env, pop_name, bcast_template=False):
    """
    :param pop_name: str
    """
    if pop_name in env.template_dict:
        return env.template_dict[pop_name]
    rank = env.comm.Get_rank()
    if not (pop_name in env.celltypes):
        raise KeyError('load_cell_templates: unrecognized cell population: %s' % pop_name)
    
    template_name = env.celltypes[pop_name]['template']
    if 'template file' in env.celltypes[pop_name]:
        template_file = env.celltypes[pop_name]['template file']
    else:
        template_file = None
    if not hasattr(h, template_name):
        find_template(env, template_name, template_file=template_file, path=env.template_paths, bcast_template=bcast_template)
    assert (hasattr(h, template_name))
    template_class = getattr(h, template_name)
    env.template_dict[pop_name] = template_class
    return template_class



def find_template(env, template_name, path=['templates'], template_file=None, bcast_template=False, root=0):
    """
    Finds and loads a template located in a directory within the given path list.
    :param env: :class:'Env'
    :param template_name: str; name of hoc template
    :param path: list of str; directories to look for hoc template
    :param template_file: str; file_name containing definition of hoc template
    :param root: int; MPI.COMM_WORLD.rank
    """
    if env.comm is None:
        bcast_template = False
    rank = env.comm.rank if env.comm is not None else 0
    found = False
    template_path = ''
    if template_file is None:
        template_file = '%s.hoc' % template_name
    if bcast_template:
        env.comm.barrier()
    if (env.comm is None) or (not bcast_template) or (bcast_template and (rank == root)):
        for template_dir in path:
            if template_file is None:
                template_path = '%s/%s.hoc' % (template_dir, template_name)
            else:
                template_path = '%s/%s' % (template_dir, template_file)
            found = os.path.isfile(template_path)
            if found and (rank == 0):
                logger.info('Loaded %s from %s' % (template_name, template_path))
                break
    if bcast_template:
        found = env.comm.bcast(found, root=root)
        env.comm.barrier()
    if found:
        if bcast_template:
            template_path = env.comm.bcast(template_path, root=root)
            env.comm.barrier()
        h.load_file(template_path)
    else:
        raise Exception('find_template: template %s not found: file %s; path is %s' %
                        (template_name, template_file, str(path)))


def configure_hoc_env(env, bcast_template=False):
    """
    :param env: :class:'Env'
    """
    h.load_file("stdrun.hoc")
    h.load_file("loadbal.hoc")
    for template_dir in env.template_paths:
        path = "%s/rn.hoc" % template_dir
        if os.path.exists(path):
            h.load_file(path)
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h('objref pc, nc, nil')
    h('strdef dataset_path')
    if hasattr(env, 'dataset_path'):
        h.dataset_path = env.dataset_path if env.dataset_path is not None else ""
    if env.use_coreneuron:
        from neuron import coreneuron
        coreneuron.enable = True
        coreneuron.verbose = 1 if env.verbose else 0
    h.pc = h.ParallelContext()
    h.pc.gid_clear()
    env.pc = h.pc
    h.dt = env.dt
    h.tstop = env.tstop
    env.t_vec = h.Vector()  # Spike time of all cells on this host
    env.id_vec = h.Vector()  # Ids of spike times on this host
    env.t_rec = h.Vector() # Timestamps of intracellular traces on this host
    if 'celsius' in env.globals:
        h.celsius = env.globals['celsius']
    ## more accurate integration of synaptic discontinuities
    if hasattr(h, 'nrn_netrec_state_adjust'):
        h.nrn_netrec_state_adjust = 1
    ## sparse parallel transfer
    if hasattr(h, 'nrn_sparse_partrans'):
        h.nrn_sparse_partrans = 1


        

def interplocs(sec):
    """Computes interpolants for xyz coords of locations in a section whose topology & geometry are defined by pt3d data.
    Based on code by Ted Carnevale.
    """

    nn = sec.n3d()

    xx = h.Vector(nn)
    yy = h.Vector(nn)
    zz = h.Vector(nn)
    dd = h.Vector(nn)
    ll = h.Vector(nn)
    
    for ii in range(0, nn):
        xx.x[ii] = sec.x3d(ii)
        yy.x[ii] = sec.y3d(ii)
        zz.x[ii] = sec.z3d(ii)
        dd.x[ii] = sec.diam3d(ii)
        ll.x[ii] = sec.arc3d(ii)
        
    ## normalize length
    ll.div(ll.x[nn - 1])

    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    dd = np.array(dd)
    ll = np.array(ll)

    u, indices = np.unique(ll, return_index=True)                                                           
    indices = np.asarray(indices)                                                                           
    if len(u) < len(ll):                                                                                    
        ll = ll[indices]                                                                                    
        xx = xx[indices]                                                                                    
        yy = yy[indices]                                                                                    
        zz = zz[indices]                                                                                    
        dd = dd[indices]                                                                                    
 
    pch_x = interpolate.pchip(ll, xx)
    pch_y = interpolate.pchip(ll, yy)
    pch_z = interpolate.pchip(ll, zz)
    pch_diam = interpolate.pchip(ll, dd)

    return pch_x, pch_y, pch_z, pch_diam


def make_rec(recid, population, gid, cell, sec=None, loc=None, ps=None, param='v', label=None, dt=None, description=''):
    """
    Makes a recording vector for the specified quantity in the specified section and location.

    :param recid: str
    :param population: str
    :param gid: integer
    :param cell: :class:'BiophysCell'
    :param sec: :class:'HocObject'
    :param loc: float
    :param ps: :class:'HocObject'
    :param param: str
    :param dt: float
    :param ylabel: str
    :param description: str
    """
    vec = h.Vector()
    if (sec is None) and (loc is None) and (ps is not None):
        hocobj = ps
        seg = ps.get_segment()
        if seg is not None:
            loc = seg.x
            sec = seg.sec
            origin = list(cell.soma)[0]
            distance = h.distance(origin(0.5), seg)
            ri = h.ri(loc, sec=sec)
        else:
            distance = None
            ri = None
    elif (sec is not None) and (loc is not None):
        hocobj = sec(loc)
        if cell.soma.__class__.__name__.lower() == "section":
            origin = cell.soma
        else:
            origin = list(cell.soma)[0]
        h.distance(sec=origin)
        distance = h.distance(loc, sec=sec)
        ri = h.ri(loc, sec=sec)
    else:
        raise RuntimeError('make_rec: either sec and loc or ps must be specified')
    section_index = None
    if sec is not None:
        for i, this_section in enumerate(cell.sections):
            if this_section == sec:
                section_index = i
                break
    if label is None:
        label = param
    if dt is None:
        vec.record(getattr(hocobj, f'_ref_{param}'))
    else:
        vec.record(getattr(hocobj, f'_ref_{param}'), dt)
    rec_dict = {'name': recid,
                'gid': gid,
                'cell': cell,
                'population': population,
                'loc': loc,
                'section': section_index,
                'distance': distance,
                'ri': ri,
                'description': description,
                'vec': vec,
                'label': label
                }

    return rec_dict
