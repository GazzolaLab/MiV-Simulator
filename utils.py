from __future__ import absolute_import, division
import copy, datetime, gc, itertools, logging, math, numbers, os.path, importlib
import pprint, string, sys, time, click
from builtins import input, map, next, object, range, str, zip
from collections import MutableMapping, Iterable, defaultdict, namedtuple
import numpy as np
import scipy
from scipy import sparse, signal
import yaml

        
is_interactive = bool(getattr(sys, 'ps1', sys.flags.interactive))

    
class Struct(object):
    def __init__(self, **items):
        self.__dict__.update(items)

    def update(self, items):
        self.__dict__.update(items)

    def __call__(self):
        return self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return f'Struct({self.__dict__})'

    def __str__(self):
        return f'<Struct>'

class ExprClosure(object):
    """
    Representation of a sympy expression with a mutable local environment.
    """
    def __init__(self, parameters, expr, consts=None, formals=None):
        self.sympy = importlib.import_module('sympy')
        self.sympy_parser = importlib.import_module('sympy.parsing.sympy_parser')
        self.sympy_abc = importlib.import_module('sympy.abc')
        self.parameters = parameters
        self.formals = formals
        if isinstance(expr, str):
            self.expr = self.sympy_parser.parse_expr(expr)
        else:
            self.expr = expr
        self.consts = {} if consts is None else consts
        self.feval = None
        self.__init_feval__()
        
    def __getitem__(self, key):
        return self.consts[key]

    def __setitem__(self, key, value):
        self.consts[key] = value
        self.feval = None
    
    def __init_feval__(self):
        fexpr = self.expr
        for k, v in viewitems(self.consts):
            sym = self.sympy.Symbol(k)
            fexpr = fexpr.subs(sym, v)
        if self.formals is None:
            formals = [self.sympy.Symbol(p) for p in self.parameters]
        else:
            formals = [self.sympy.Symbol(p) for p in self.formals]
        self.feval = self.sympy.lambdify(formals, fexpr, "numpy")

    def __call__(self, *x):
        if self.feval is None:
            self.__init_feval__()
        return self.feval(*x)

    def __repr__(self):
        return f'ExprClosure(expr: {self.expr} formals: {self.formals} parameters: {self.parameters} consts: {self.consts})'

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        deepcopy_fields = ['parameters', 'formals', 'consts', 'expr']
        for k in deepcopy_fields:
            v = self.__dict__[k]
            setattr(result, k, copy.deepcopy(v, memo))
        for k, v in self.__dict__.items():
            if k not in deepcopy_fields:
                setattr(result, k, v)
        result.__init_feval__()
        memo[id(self)] = result
        return result

    
class Promise(object):
    """
    An object that represents a closure and unapplied arguments.
    """
    def __init__(self, clos, args):
        assert(isinstance(clos, ExprClosure))
        self.clos = clos
        self.args = args
    def __repr__(self):
        return f'Promise(clos: {self.clos} args: {self.args})'
    def append(self, arg):
        self.args.append(arg)
    


class Context(object):
    """
    A container replacement for global variables to be shared and modified by any function in a module.
    """

    def __init__(self, namespace_dict=None, **kwargs):
        self.update(namespace_dict, **kwargs)

    def update(self, namespace_dict=None, **kwargs):
        """
        Converts items in a dictionary (such as globals() or locals()) into context object internals.
        :param namespace_dict: dict
        """
        if namespace_dict is not None:
            self.__dict__.update(namespace_dict)
        self.__dict__.update(kwargs)

    def __call__(self):
        return self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return f'Context({self.__dict__})'

    def __str__(self):
        return f'<Context>'
    
    
class RunningStats(object):

    def __init__(self):
        self.n = 0
        self.m1 = 0.
        self.m2 = 0.
        self.m3 = 0.
        self.m4 = 0.
        self.min = float('inf')
        self.max = float('-inf')
        
    def clear(self):
        self.n = 0
        self.m1 = 0.
        self.m2 = 0.
        self.m3 = 0.
        self.m4 = 0.
        self.min = float('inf')
        self.max = float('-inf')
        

    def update(self, x):
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        n1 = self.n
        self.n += 1
        n = self.n
        delta = x - self.m1
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        self.m1 += delta_n
        self.m4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * self.m2 - 4 * delta_n * self.m3
        self.m3 += term1 * delta_n * (n - 2) - 3 * delta_n * self.m2
        self.m2 += term1

    def mean(self):
        return self.m1

    def variance(self):
        return self.m2 / (self.n - 1.0)

    def standard_deviation(self):
        return math.sqrt(self.variance())

    def skewness(self):
        return math.sqrt(self.n) * self.m3 / (self.m2 ** 1.5)

    def kurtosis(self):
        return self.n * self.m4 / (self.m2*self.m2) - 3.0
        
    
    @classmethod
    def combine(cls, a, b):
        combined = cls()
        
        combined.n = a.n + b.n
        combined.min = min(a.min, b.min)
        combined.max = max(a.max, b.max)
    
        delta = b.m1 - a.m1;
        delta2 = delta*delta;
        delta3 = delta*delta2;
        delta4 = delta2*delta2;
    
        combined.m1 = (a.n*a.m1 + b.n*b.m1) / combined.n;
        
        combined.m2 = a.m2 + b.m2 + delta2 * a.n * b.n / combined.n
    
        combined.m3 = a.m3 + b.m3 + delta3 * a.n * b.n * (a.n - b.n)/(combined.n*combined.n)
        combined.m3 += 3.0*delta * (a.n*b.m2 - b.n*a.m2) / combined.n
    
        combined.m4 = a.m4 + b.m4 + delta4*a.n*b.n * (a.n*a.n - a.n*b.n + b.n*b.n) / \
          (combined.n*combined.n*combined.n)
        combined.m4 += 6.0*delta2 * (a.n*a.n*b.m2 + b.n*b.n*a.m2)/(combined.n*combined.n) + \
          4.0*delta*(a.n*b.m3 - b.n*a.m3) / combined.n
    
        return combined

## https://github.com/pallets/click/issues/605
class EnumChoice(click.Choice):
    def __init__(self, enum, case_sensitive=False, use_value=False):
        self.enum = enum
        self.use_value = use_value
        choices = [str(e.value) if use_value else e.name for e in self.enum]
        super().__init__(choices, case_sensitive)

    def convert(self, value, param, ctx):
        if value in self.enum:
            return value
        result = super().convert(value, param, ctx)
        # Find the original case in the enum
        if not self.case_sensitive and result not in self.choices:
            result = next(c for c in self.choices if result.lower() == c.lower())
        if self.use_value:
            return next(e for e in self.enum if str(e.value) == result)
        return self.enum[result]

class IncludeLoader(yaml.Loader):
    """
    YAML loader with `!include` handler.
    """

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        yaml.Loader.__init__(self, stream)

    def include(self, node):
        """

        :param node:
        :return:
        """
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, IncludeLoader)


IncludeLoader.add_constructor('!include', IncludeLoader.include)

class ExplicitDumper(yaml.SafeDumper):
    """
    YAML dumper that will never emit aliases.
    """

    def ignore_aliases(self, data):
        return True

def config_logging(verbose):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)


def get_root_logger():
    logger = logging.getLogger('biophys_microcircuit')
    return logger


def get_module_logger(name):
    logger = logging.getLogger('%s' % name)
    return logger


def get_script_logger(name):
    logger = logging.getLogger('biophys_microcircuit.%s' % name)
    return logger


# This logger will inherit its settings from the root logger, created in biophys_microcircuit.env
logger = get_module_logger(__name__)


def write_to_yaml(file_path, data, convert_scalars=False):
    """

    :param file_path: str (should end in '.yaml')
    :param data: dict
    :param convert_scalars: bool
    :return:
    """
    with open(file_path, 'w') as outfile:
        if convert_scalars:
            data = nested_convert_scalars(data)
        yaml.dump(data, outfile, default_flow_style=False, Dumper=ExplicitDumper)


def read_from_yaml(file_path, include_loader=None):
    """

    :param file_path: str (should end in '.yaml')
    :return:
    """
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            if include_loader is None:
                Loader = yaml.FullLoader
            else:
                Loader = include_loader
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise IOError('read_from_yaml: invalid file_path: %s' % file_path)


def print_param_dict_like_yaml(param_dict, digits=6):
    """
    Assumes a flat dict with int or float values.
    :param param_dict: dict
    :param digits: int
    """
    for param_name, param_val in viewitems(param_dict):
        if isinstance(param_val, int):
            print('%s: %s' % (param_name, param_val))
        else:
            print('%s: %.*E' % (param_name, digits, param_val))


def nested_convert_scalars(data):
    """
    Crawls a nested dictionary, and converts any scalar objects from numpy types to python types.
    :param data: dict
    :return: dict
    """
    if isinstance(data, dict):
        for key in data:
            data[key] = nested_convert_scalars(data[key])
    elif isinstance(data, Iterable) and not isinstance(data, (str, tuple)):
        data = list(data)
        for i in range(len(data)):
            data[i] = nested_convert_scalars(data[i])
    elif hasattr(data, 'item'):
        data = data.item()
    return data

def is_iterable(obj):
    return isinstance(obj, Iterable)

def list_index(element, lst):
    """

    :param element:
    :param lst:
    :return:
    """
    try:
        index_element = lst.index(element)
        return index_element
    except ValueError:
        return None


def list_find(f, lst):
    """

    :param f:
    :param lst:
    :return:
    """
    i = 0
    for x in lst:
        if f(x):
            return i
        else:
            i = i + 1
    return None


def list_find_all(f, lst):
    """

    :param f:
    :param lst:
    :return:
    """
    i = 0
    res = []
    for i, x in enumerate(lst):
        if f(x):
            res.append(i)
    return res


def list_argsort(f, seq):
    """
    http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
    lambda version by Tony Veijalainen
    :param f:
    :param seq:
    :return:
    """
    return [i for i, x in sorted(enumerate(seq), key=lambda x: f(x[1]))]



def viewattrs(obj):
    if hasattr(obj, 'n_sequence_fields'):
        return dir(obj)[:obj.n_sequence_fields]
    else:
        return vars(obj)

def viewitems(obj, **kwargs):
    """
    Function for iterating over dictionary items with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewitems", None)
    if func is None:
        func = obj.items
    return func(**kwargs)


def viewkeys(obj, **kwargs):
    """
    Function for iterating over dictionary keys with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewkeys", None)
    if func is None:
        func = obj.keys
    return func(**kwargs)


def viewvalues(obj, **kwargs):
    """
    Function for iterating over dictionary values with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewvalues", None)
    if func is None:
        func = obj.values
    return func(**kwargs)


def zip_longest(*args, **kwds):
    if hasattr(itertools, 'izip_longest'):
        return itertools.izip_longest(*args, **kwds)
    else:
        return itertools.zip_longest(*args, **kwds)


def consecutive(data):
    """
    Returns a list of arrays with consecutive values from data.
    """
    return np.split(data, np.where(np.diff(data) != 1)[0]+1)


def ifilternone(iterable):
    for x in iterable:
        if not (x is None):
            yield x


def flatten(iterables):
    return (elem for iterable in ifilternone(iterables) for elem in iterable)

def imapreduce(iterable, fmap, freduce, init=None):
    it = iter(iterable)
    if init is None:
        value = fmap(next(it))
    else:
        value = init
    for x in it:
        value = freduce(value, fmap(x))
    return value
        

def make_geometric_graph(x, y, z, edges):
    """ Builds a NetworkX graph with xyz node coordinates and the node indices
        of the end nodes.

        Parameters
        -----------
        x: ndarray
            x coordinates of the points
        y: ndarray
            y coordinates of the points
        z: ndarray
            z coordinates of the points
        edges: the (2, N) array returned by compute_delaunay_edges()
            containing node indices of the end nodes. Weights are applied to
            the edges based on their euclidean length for use by the MST
            algorithm.

        Returns
        ---------
        g: A NetworkX undirected graph

        Notes
        ------
        We don't bother putting the coordinates into the NX graph.
        Instead the graph node is an index to the column.
    """
    import networkx as nx
    xyz = np.array((x, y, z))

    def euclidean_dist(i, j):
        d = xyz[:, i] - xyz[:, j]
        return np.sqrt(np.dot(d, d))

    g = nx.Graph()
    for i, j in edges:
        g.add_edge(i, j, weight=euclidean_dist(i, j))
    return g


def random_choice_w_replacement(ranstream, n, p):
    """

    :param ranstream:
    :param n:
    :param p:
    :return:
    """
    return ranstream.multinomial(n, p.ravel())


def make_random_clusters(centers, n_samples_per_center, n_features=2, cluster_std=1.0, center_ids=None,
                         center_box=(-10.0, 10.0), random_seed=None):
    """Generate isotropic Gaussian blobs for clustering.

    Parameters
    ----------
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
    n_samples_per_center : int array
        Number of points for each cluster.
    n_features : int, optional (default=2)
        The number of features for each sample.
    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_ids : array of integer center ids, if None then centers will be numbered 0 .. n_centers-1
    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    random_seed : int or None, optional (default=None)
        If int, random_seed is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    Examples
    --------
    >>> X, y = make_random_clusters (centers=6, n_samples_per_center=np.array([1,3,10,15,7,9]), n_features=1, \
                                     center_ids=np.array([10,13,21,25,27,29]).reshape(-1,1), cluster_std=1.0, \
                                     center_box=(-10.0, 10.0))
    >>> print(X.shape)
    (45, 1)
    >>> y
    array([10, 13, 13, 13, ..., 29, 29, 29])
    """
    rng = np.random.RandomState(random_seed)

    if isinstance(centers, numbers.Integral):
        centers = np.sort(rng.uniform(center_box[0], center_box[1], \
                                      size=(centers, n_features)), axis=0)
    else:
        assert (isinstance(centers, np.ndarray))
        n_features = centers.shape[1]

    if center_ids is None:
        center_ids = np.arange(0, centers.shape[0])

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centers)) * cluster_std

    X = []
    y = []

    n_centers = centers.shape[0]

    for i, (cid, n, std) in enumerate(zip(center_ids, n_samples_per_center, cluster_std)):
        if n > 0:
            X.append(centers[i] + rng.normal(scale=std, size=(n, n_features)))
            y += [cid] * n

    X = np.concatenate(X)
    y = np.array(y)

    return X, y


def random_clustered_shuffle(centers, n_samples_per_center, center_ids=None, cluster_std=1.0, center_box=(-1.0, 1.0),
                             random_seed=None):
    """Generates a Gaussian random clustering given a number of cluster
    centers, samples per each center, optional integer center ids, and
    cluster standard deviation.

    Parameters
    ----------
    centers : int or array of shape [n_centers]
        The number of centers to generate, or the fixed center locations.
    n_samples_per_center : int array
        Number of points for each cluster.
    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_ids : array of integer center ids, if None then centers will be numbered 0 .. n_centers-1
    random_seed : int or None, optional (default=None)
        If int, random_seed is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    >>> x = random_clustered_shuffle(centers=6,center_ids=np.array([10,13,21,25,27,29]).reshape(-1,1), \
                                     n_samples_per_center=np.array([1,3,10,15,7,9]))
    >>> array([10, 13, 13, 25, 13, 29, 21, 25, 27, 21, 27, 29, 25, 25, 25, 21, 29,
               27, 25, 21, 29, 25, 25, 25, 25, 29, 21, 25, 21, 29, 29, 29, 21, 25,
               29, 21, 27, 27, 21, 27, 25, 21, 25, 27, 25])
    """

    if isinstance(centers, numbers.Integral):
        n_centers = centers
    else:
        assert (isinstance(centers, np.ndarray))
        n_centers = len(centers)

    X, y = make_random_clusters(centers, n_samples_per_center, n_features=1, \
                                center_ids=center_ids, cluster_std=cluster_std, center_box=center_box, \
                                random_seed=random_seed)
    s = np.argsort(X, axis=0).ravel()
    return y[s].ravel()


def rejection_sampling(gen, n, clip):
    if clip is None:
        result = gen(n)
    else:
        clip_min, clip_max = clip
        remaining = n
        samples = []
        while remaining > 0:
            sample = gen(remaining)
            filtered = sample[np.where((sample >= clip_min) & (sample <= clip_max))]
            samples.append(filtered)
            remaining -= len(filtered)
        result = np.concatenate(tuple(samples))

    return result


def NamedTupleWithDocstring(docstring, *ntargs):
    """
    A convenience wrapper to add docstrings to named tuples. This is only needed in
    python 2, where __doc__ is not writeable.
    https://stackoverflow.com/questions/1606436/adding-docstrings-to-namedtuples
    """
    nt = namedtuple(*ntargs)

    class NT(nt):
        __doc__ = docstring
        __slots__ = ()  ## disallow mutable slots in order to keep performance advantage of tuples

    return NT


def partitionn(items, predicate=int, n=2):
    """
    Filter an iterator into N parts lazily
    http://paddy3118.blogspot.com/2013/06/filtering-iterator-into-n-parts-lazily.html
    """
    tees = itertools.tee(((predicate(item), item)
                          for item in items), n)
    return ((lambda i: (item for pred, item in tees[i] if pred == i))(x)
            for x in range(n))


def generator_peek(iterable):
    """
    If the iterable is empty, return None, otherwise return a tuple with the
    first element and the iterable with the first element attached back.
    """
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)


def generator_ifempty(iterable):
    """
    If the iterable is empty, return None, otherwise return the
    iterable with the first element attached back.
    """
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return itertools.chain([first], iterable)


def compose_iter(f, it):
    """
    Given a function and an iterator, apply the function to
    each element in the iterator and return the element.
    """
    for x in it:
        f(x)
        yield x


def profile_memory(logger):
    from guppy import hpy
    hprof = hpy()
    logger.info(hprof.heap())


def update_bins(bins, binsize, *xs):
    idxs = tuple(math.floor(x / binsize) for x in xs)
    if idxs in bins:
        bins[idxs] += 1
    else:
        bins[idxs] = 1


def finalize_bins(bins, binsize):
    bin_keys = zip_longest(*viewkeys(bins))
    bin_ranges = [(int(min(ks)), int(max(ks))) for ks in bin_keys]
    dims = tuple((imax - imin + 1) for imin, imax in bin_ranges)
    if len(dims) > 1:
        grid = sparse.dok_matrix(dims, dtype=np.int)
    else:
        grid = np.zeros(dims)
    bin_edges = [[binsize * k for k in range(imin, imax + 1)] for imin, imax in bin_ranges]
    for i in bins:
        idx = tuple([int(ii - imin) for ii, (imin, imax) in zip(i, bin_ranges)])
        grid[idx] = bins[i]
    result = tuple([grid] + [np.asarray(edges) for edges in bin_edges])
    return result


def merge_bins(bins1, bins2, datatype):
    for i, count in viewitems(bins2):
        if i in bins1:
            bins1[i] += count
        else:
            bins1[i] = count
    return bins1


def add_bins(bins1, bins2, datatype):
    for item in bins2:
        if item in bins1:
            bins1[item] += bins2[item]
        else:
            bins1[item] = bins2[item]
    return bins1



def baks(spktimes, time, a=1.5, b=None):
    """
    Bayesian Adaptive Kernel Smoother (BAKS)
    BAKS is a method for estimating firing rate from spike train data that uses kernel smoothing technique 
    with adaptive bandwidth determined using a Bayesian approach
    ---------------INPUT---------------
    - spktimes : spike event times [s]
    - time : time points at which the firing rate is estimated [s]
    - a : shape parameter (alpha) 
    - b : scale parameter (beta)
    ---------------OUTPUT---------------
    - rate : estimated firing rate [nTime x 1] (Hz)
    - h : adaptive bandwidth [nTime x 1]

    Based on "Estimation of neuronal firing rate using Bayesian adaptive kernel smoother (BAKS)"
    https://github.com/nurahmadi/BAKS
    """
    from scipy.special import gamma

    n = len(spktimes)
    sumnum = 0
    sumdenom = 0

    if b is None:
        b = 0.42
    b = float(n) ** b

    for i in range(n):
        numerator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a)
        denominator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a - 0.5)
        sumnum = sumnum + numerator
        sumdenom = sumdenom + denominator

    h = (gamma(a) / gamma(a + 0.5)) * (sumnum / sumdenom)
    rate = np.zeros((len(time),))
    for j in range(n):
        x = np.asarray(-((time - spktimes[j]) ** 2) / (2. * h ** 2), dtype=np.float128)
        K = (1. / (np.sqrt(2. * np.pi) * h)) * np.exp(x)
        rate = rate + K

    return rate, h



def get_R2(y_test, y_pred):

    """
    Obtain coefficient of determination (R-squared, R2)

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of features)
    y_pred - the predicted outputs (a matrix of size number of examples x number of features)

    Returns
    -------
    An array of R2s for each feature
    """

    R2_list=[]
    for i in range(y_test.shape[1]):
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2)
    R2_array=np.array(R2_list)
    return R2_array


def mvcorrcoef(X,y):
    """
    Multivariate correlation coefficient.
    """
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum(np.multiply(X-Xm,y-ym),axis=1)
    r_den = np.sqrt(np.sum(np.square(X-Xm),axis=1)*np.sum(np.square(y-ym)))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.true_divide(r_num, r_den)
        r[r == np.inf] = 0
        r = np.nan_to_num(r)
    return r


def autocorr (y, lag):
    leny = y.shape[1]
    a = y[0,0:leny-lag].reshape(-1)
    b = y[0,lag:leny].reshape(-1)
    m = np.vstack((a[0,:].reshape(-1), b[0,:].reshape(-1)))
    r = np.corrcoef(m)[0,1]
    if math.isnan(r):
        return 0.
    else:
        return r

def butter_bandpass_filter(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos


def apply_filter(data, sos):
    y = signal.sosfiltfilt(sos, data)
    return y


def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1, A=1.):
    ## prevent exp underflow/overflow
    exparg = np.clip(((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)), -500., 500.)
    return A * np.exp(-exparg)

def gaussian(x, mu, sig, A=1.):
    return A * np.exp(-np.power(x - mu, 2.) / (2. * np.power(sig, 2.)))

    
def get_low_pass_filtered_trace(trace, t, down_dt=0.5):
    import scipy.signal as signal
    down_t = np.arange(np.min(t), np.max(t), down_dt)
    # 2000 ms Hamming window, ~3 Hz low-pass filter
    window_len = int(2000./down_dt)
    pad_len = int(window_len / 2.)
    ramp_filter = signal.firwin(window_len, 2., nyq=1000. / 2. / down_dt)
    down_sampled = np.interp(down_t, t, trace)
    padded_trace = np.zeros(len(down_sampled) + window_len)
    padded_trace[pad_len:-pad_len] = down_sampled
    padded_trace[:pad_len] = down_sampled[::-1][-pad_len:]
    padded_trace[-pad_len:] = down_sampled[::-1][:pad_len]
    down_filtered = signal.filtfilt(ramp_filter, [1.], padded_trace, padlen=pad_len)
    down_filtered = down_filtered[pad_len:-pad_len]
    filtered = np.interp(t, down_t, down_filtered)
    return filtered


def get_trial_time_ranges(time_vec, n_trials, t_offset=0.):
    time_vec = np.asarray(time_vec, dtype=np.float32) - t_offset
    t_trial = (np.max(time_vec) - np.min(time_vec)) / float(n_trials)
    t_start = np.min(time_vec)
    t_trial_ranges = [ (float(i)*t_trial + t_start, float(i)*t_trial + t_start + t_trial) for i in range(n_trials) ] 
    return t_trial_ranges


def get_trial_time_indices(time_vec, n_trials, t_offset=0.):
    time_vec = np.asarray(time_vec, dtype=np.float32) - t_offset
    t_trial = (np.max(time_vec) - np.min(time_vec)) / float(n_trials)
    t_start = np.min(time_vec)
    t_trial_ranges = [ (float(i)*t_trial + t_start, float(i)*t_trial + t_start + t_trial) for i in range(n_trials) ] 
    t_trial_inds = [ np.where((time_vec >= (t_trial_start + t_offset)) & (time_vec < t_trial_end))[0]
                     for t_trial_start, t_trial_end in t_trial_ranges ]
    return t_trial_inds


def contiguous_ranges(condition, return_indices=False):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a list of ranges with the start and end index of each region. Code based on:
    https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array/4495197
    """

    # Find the indices of changes in "condition"
    d = np.diff(condition)
    ranges, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the ranges by 1 to the right.
    ranges += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        ranges = np.r_[0, ranges]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        ranges = np.r_[ranges, condition.size] # Edit

    # Reshape the result into two columns
    ranges.shape = (-1,2)

    if return_indices:
        result = ( np.arange(*r) for r in ranges )
    else:
        result = ranges

    return result


def signal_power_spectrogram(signal, fs, window_size, window_overlap):
    """
    Computes the power spectrum of the specified signal.
    
    A Hanning window with the specified size and overlap is used.
    
    Parameters
    ----------
    signal: numpy.ndarray
        The input signal
    fs: int
        Sampling frequency of the input signal
    window_size: int
        Size of the Hann windows in samples
    window_overlap: float
        Overlap between Hann windows as fraction of window_size

    Returns
    -------
    f: numpy.ndarray
        Array of frequency values for the first axis of the returned spectrogram
    t: numpy.ndarray
        Array of time values for the second axis of the returned spectrogram
    sxx: numpy.ndarray
        Power spectrogram of the input signal with axes [frequency, time]
    """
    from scipy.signal import spectrogram, get_window

    nperseg    = window_size
    win        = get_window('hanning', nperseg)
    noverlap   = int(window_overlap * nperseg)

    f, t, sxx = spectrogram(x=signal, fs=fs, window=win, noverlap=noverlap, mode="psd")

    return f, t, sxx

def signal_psd (s, Fs, frequency_range=(0,500), window_size=4096, overlap=0.9):
    
    nperseg    = window_size
    win        = signal.get_window('hanning', nperseg)
    noverlap   = int(overlap * nperseg)

    freqs, psd = signal.welch(s, fs=Fs, scaling='density', nperseg=nperseg, noverlap=noverlap,
                              window=win, return_onesided=True)

    freqinds = np.where((freqs >= frequency_range[0]) & (freqs <= frequency_range[1]))

    freqs = freqs[freqinds]
    psd = psd[freqinds]
    if np.all(psd):
        psd = 10. * np.log10(psd)

    peak_index = np.where(psd == np.max(psd))[0]

    return psd, freqs, peak_index


def baks(spktimes, time, a=1.5, b=None):
    """
    Bayesian Adaptive Kernel Smoother (BAKS)
    BAKS is a method for estimating firing rate from spike train data that uses kernel smoothing technique 
    with adaptive bandwidth determined using a Bayesian approach
    ---------------INPUT---------------
    - spktimes : spike event times [s]
    - time : time points at which the firing rate is estimated [s]
    - a : shape parameter (alpha) 
    - b : scale parameter (beta)
    ---------------OUTPUT---------------
    - rate : estimated firing rate [nTime x 1] (Hz)
    - h : adaptive bandwidth [nTime x 1]

    Based on "Estimation of neuronal firing rate using Bayesian adaptive kernel smoother (BAKS)"
    https://github.com/nurahmadi/BAKS
    """
    from scipy.special import gamma

    n = len(spktimes)
    sumnum = 0
    sumdenom = 0

    if b is None:
        b = 0.42
    b = float(n) ** b

    for i in range(n):
        numerator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a)
        denominator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a - 0.5)
        sumnum = sumnum + numerator
        sumdenom = sumdenom + denominator

    h = (gamma(a) / gamma(a + 0.5)) * (sumnum / sumdenom)
    rate = np.zeros((len(time),))
    for j in range(n):
        x = np.asarray(-((time - spktimes[j]) ** 2) / (2. * h ** 2), dtype=np.float128)
        K = (1. / (np.sqrt(2. * np.pi) * h)) * np.exp(x)
        rate = rate + K

    return rate, h
