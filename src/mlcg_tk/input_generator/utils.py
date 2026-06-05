import pandas as pd
from typing import List, Optional, Union, Tuple, Dict, Literal
import numpy as np
import mdtraj as md
import MDAnalysis as mda
import warnings
from tqdm import tqdm
from numbers import Integral
from pathlib import Path
from scipy.sparse import sparray, csr_array, issparse, eye
from MDAnalysis.coordinates.TRR import TRRWriter
from functools import wraps

import aggforce  # needed for monkey-patching
from aggforce import (
    LinearMap,
    guess_pairwise_constraints,
    project_forces,
    constraint_aware_uni_map,
    qp_linear_map,
)
from aggforce.map.core import _has_nans

# monkey patches to aggforce: _LinearMap supporting sparse arrays
class _LinearMap(LinearMap):
    def __init__(
        self,
        mapping : Union[List[List[int]], csr_array, np.ndarray],
        n_fg_sites : Union[int, None]=None,
        handle_nans : Union[bool, Literal["safe"]]=True,
        nan_check_threshold : float=1e-6
    ) -> None:
        if (issparse(mapping) or isinstance(mapping, np.ndarray)
            ) and len(mapping.shape) == 2:
            if n_fg_sites is not None:
                raise ValueError(
                    "Cannot specify n_fg_sites when mapping is ArrayLike. "
                    "Let it be inferred."
                )
            self._standard_matrix = mapping
            if len(self.standard_matrix.shape) != 2:
                raise ValueError(
                    f"mapping ({mapping}) does not cast into a 2-d numpy array."
            )
        elif hasattr(mapping, "__iter__"):
            # assume we are in the case of iterable of lists
            if n_fg_sites is None:
                raise ValueError()
            mapping = list(mapping)
            n_cg_sites = len(mapping)
            mapping_mat = np.zeros((n_cg_sites, n_fg_sites))
            for site, site_contents in enumerate(mapping):
                local_map = np.zeros(n_fg_sites)
                local_map[site_contents] = 1 / len(site_contents)
                mapping_mat[site, :] = local_map
            self._standard_matrix = mapping_mat
        else:
            raise ValueError("Cannot understanding mapping f{mapping}.")
        #
        self.handle_nans = handle_nans
        if self.handle_nans:
            if ((isinstance(self._standard_matrix, np.ndarray) and
                not np.all(np.isfinite(self._standard_matrix))) or
                (issparse(self._standard_matrix) and
                not np.all(np.isfinite(self._standard_matrix.data)))):
                raise ValueError(
                    "Nan checking can only be performed in "
                    "standard_matrix is itself finite."
                )
        self.nan_check_threshold = nan_check_threshold
    #
    def close_to_identity(self, threshold: float = 1e-8):
        # had to redefine because of a bug in the original class
        #
        matrix = self._standard_matrix
        internal_shape = matrix.shape
        if internal_shape[0] != internal_shape[1]:
            return False
        if issparse(matrix):
            return (abs(matrix - eye(internal_shape[0], format=matrix.format))
                    ).max() <= threshold
        return np.allclose(matrix, np.eye(internal_shape[0]), atol=threshold)
    #
    def __call__(
            self,
            points: np.ndarray,
    ) -> np.ndarray:
        # had to redefine because np.einsum does not work on sparse mat
        #
        # reimports/redefinitions
        def trjdot(
                points : np.ndarray, factor : np.ndarray
        ) -> np.ndarray:
            """Supports sparse arrays."""
            if factor.ndim == 2:
                if issparse(factor):
                    return np.stack(
                        [points[:, :, d] @ factor.T
                            for d in range(points.shape[2])],
                        axis=-1,
                    )
                #   
                return np.einsum(
                    "tfd,cf->tcd",
                    points,
                    factor,
                    optimize=["einsum_path", (0, 1)],
                )
            #
            if factor.ndim == 3:
                if issparse(factor):
                    raise ValueError(
                        "Sparse factor matrices are only supported for "
                        "time-independent mappings (factor.ndim == 2)."
                    )
                #
                return np.einsum(
                    "...fd,...cf->...cd",
                    points,
                    factor,
                    optimize=["einsum_path", (0, 1)],
                )
            #
            raise ValueError(
                f"Factor matrix has incompatible shape {factor.shape}."
            )
        #
        # from now on, the same function as aggmap.LinearMap.__call__
        nan_handling = self.handle_nans and _has_nans(points)
        if nan_handling:
            input_mask = np.isnan(points)
            if self.handle_nans == "safe":
                input_matrix = points.copy()
            else:
                input_matrix = points
            input_matrix[input_mask] = 0.0
            raw_result = trjdot(input_matrix, self.standard_matrix)
            input_matrix[input_mask] = -1.0
            pushed_result = trjdot(input_matrix, self.standard_matrix)
            if not np.allclose(
                raw_result, pushed_result,
                atol=self.nan_check_threshold
            ):
                raise ValueError(
                    "NaN handling is on and results seem to depend on NaN "
                    "positions in input array. Check input and standard_matrix."
                )
            input_matrix[input_mask] = np.nan
            return raw_result
        else:
            return trjdot(points, self.standard_matrix)
    #
    @property
    def T(self):
        """LinearMap defined by transpose of its standard matrix."""
        return self.__class__(
            mapping=self.standard_matrix.T,
            handle_nans=self.handle_nans,
            nan_check_threshold=self.nan_check_threshold,
        )
    #
    def __matmul__(self, lm: "LinearMap", /):
        """LinearMap defined by multiplying the standard_matrix's of arguments."""
        return self.__class__(
            mapping=self.standard_matrix @ lm.standard_matrix,
            handle_nans=self.handle_nans,
            nan_check_threshold=self.nan_check_threshold,
        )
    #
    def __rmul__(self, c: float, /):
        """LinearMap defined by multiplying the standard_matrix's with a coefficient."""
        return self.__class__(
            mapping=c * self.standard_matrix,
            handle_nans=self.handle_nans,
            nan_check_threshold=self.nan_check_threshold,
        )
    #
    def __add__(self, lm: "LinearMap", /):
        """LinearMap defined by adding standard_matrices."""
        return self.__class__(
            mapping=self.standard_matrix + lm.standard_matrix,
            handle_nans=self.handle_nans,
            nan_check_threshold=self.nan_check_threshold,
        )

# inherit docstrings & apply patch
_LinearMap.__doc__ = LinearMap.__doc__
_LinearMap.close_to_identity.__doc__ = LinearMap.close_to_identity.__doc__
_LinearMap.__init__.__doc__ = f'''{LinearMap.__init__.__doc__}

Remarks
-------
"mapping" can also by a scipy.sparse array.'''
_LinearMap.__call__.__doc__ = LinearMap.__call__.__doc__
aggforce.map.LinearMap = _LinearMap

# monkey patches to aggforce: immensely faster reduce_constraint_sets
from collections import defaultdict
def reduce_constraint_sets(constraints):
    """Merge overlapping constraint sets into disjoint components."""
    if len(constraints) <= 1:
        return set(constraints)
    #
    parent = {}
    #
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x
    #
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    #
    # initialize atoms
    for c in constraints:
        for atom in c:
            parent.setdefault(atom, atom)
    #
    # union atoms within each constraint
    for c in constraints:
        c = iter(c)
        first = next(c)
        for atom in c:
            union(first, atom)
    #
    # build connected components
    components = defaultdict(set)
    for atom in parent:
        components[find(atom)].add(atom)
    #
    return {frozenset(component) for component in components.values()}

# apply patch
aggforce.constraints.tools.reduce_constraint_sets = reduce_constraint_sets

# monkey patches to aggforce: immensely faster constraint_aware_uni_map
# which also initialize force_matrix as sparse
ForcesTrajectory = aggforce.trajectory.ForcesTrajectory
SeperableTMap = aggforce.map.SeperableTMap
Constraints = aggforce.constraints.Constraints
from itertools import product
def constraint_aware_uni_map(
    traj: ForcesTrajectory,  # noqa: ARG001
    coord_map: LinearMap,
    constraints: Union[None, Constraints] = None,
) -> SeperableTMap:
    if constraints is None:
        constraints = set()
    # get which sites have nonzero contributions to each cg site
    cg_sets = [set(np.nonzero(row)[0]) for row in coord_map.standard_matrix]
    #
    constraints = aggforce.constraints.tools.reduce_constraint_sets(constraints)
    # add atoms which are related by constraint to those already in cg sites
    # faster implementation
    # NOTE: I did myself, without those phony AIs
    cache = {}
    for group in constraints:
        for id in group:
            cache[id] = group
    for group in cg_sets:
        for id in list(group):
            group.update(cache.get(id, {}))
    #
    rows = []
    columns = []
    # place a 1 where all original or those pulled in by constraints are
    for cg_index, cg_contents in enumerate(cg_sets):
        rows += [cg_index] * len(cg_contents)
        columns += cg_contents
    # create force array as sparse matrix
    force_map_mat = csr_array(
        (np.ones(len(rows)), (rows, columns)),
        shape=coord_map.standard_matrix.shape)
    force_map = aggforce.map.LinearMap(force_map_mat)
    return SeperableTMap(coord_map=coord_map, force_map=force_map)

# apply patch
aggforce.qp.basicagg.constraint_aware_uni_map = constraint_aware_uni_map


def expand_index(key, shape):
    """
    NumPy-like indexing expansion for arbitrary dimensions.

    Parameters
    ----------
    key: int, slice, iterable
        Numpy key.
    shape: tuple
        Array's shape: x.shape.

    Returns
    -------
    indices: list of integer arrays
        For each dimension, the selected indices
    product: bool
        If `True`, need to do the Cartesian product of indices;
        if `False`, need to zip over the indices.
    out_shape : tuple
        `x[key].shape`
    """
    # process key
    if not isinstance(key, tuple):
        key = (key,)
    #
    # determine output shape
    base_dim = (1,) * len(shape)
    small_array = np.zeros(base_dim, dtype=np.int8)
    virtual_array = np.broadcast_to(small_array, shape)
    out_shape = virtual_array[key].shape
    #
    # determine indices
    indices = []
    shape = list(shape)
    for k in key:
        # new axis (does not consume shape axis)
        if k is None:
            continue
        #
        # get axis size
        n = shape.pop(0)
        #
        # slice
        if isinstance(k, slice):
            indices.append(np.arange(*k.indices(n)))
            continue
        #
        # list, tuple, range, np.ndarray
        if isinstance(k, (list, tuple, range, np.ndarray)):
            indices.append(np.array(k, dtype=int))
            continue
        #
        # integral
        if isinstance(k, Integral):
            indices.append(np.array([k]))
            continue
        #
        raise TypeError(f"Unsupported index type: {type(k)}")
    #
    # consume remaining axes if key was shorter
    while shape:
        indices.append(np.arange(shape.pop(0)))
    #
    # determine wether you need the cartsian "product"
    product = np.prod([len(idxs) for idxs in indices]) == np.prod(out_shape)
    #
    return indices, product, out_shape


class DataIterable:
    """
    Helper class for loading positions and forces from a MDAnalysis
    trajectory.
    """
    def __init__(self,
                 trajectory,
                 atoms,
                 attribute='positions',
                 scaling=1.0,
                 precision=np.float32):
        self.trajectory = trajectory
        self.atoms = atoms
        self.attribute = attribute
        self.scaling = scaling
        self.precision = precision
    def _get(self):
        return getattr(self.atoms, self.attribute) / self.scaling
    def __getitem__(self, key):
        """
        Extract data from iterable and return the result as a numpy array.
        The same indexing as numpy applies: `self[key] = self[:][key]`,
        where `self[:]` are the full data, which may be too large to be
        loaded on memory. 
        """
        indices, product, out_shape = expand_index(key, self.shape)
        if product:
            temp_shape = tuple(len(idxs) for idxs in indices)
            result = np.zeros(temp_shape, dtype=self.precision)
            for i, frame in enumerate(indices[0]):
                self.trajectory[frame]
                result[i] = self._get()[indices[1]][:, indices[2]]
        else:
            result = np.zeros(len(indices[0]), dtype=self.precision)
            for i, (frame, j, k) in enumerate(zip(*indices)):
                self.trajectory[frame]
                result[i] = self._get()[j, k]
        return result.reshape(out_shape)
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    def __len__(self):
        return len(self.trajectory)
    def __repr__(self):
        return f'DataIterable of shape {self.shape}'
    @property
    def n_atoms(self):
        return len(self.atoms)
    @property
    def shape(self):
        return len(self), self.n_atoms, 3


from .prior_gen import PriorBuilder


def cg_matmul(
    map_arr: Union[np.ndarray, sparray], timeseries_arr: np.ndarray
) -> np.ndarray:
    r"""Function to perform array multiplication for both numpy and scipy sparse arrays

    Parameters
    ----------

    map_arr : Union[np.ndarray,sparray]
        array of shape (n_beads,n_atoms) representing a linear CG mapping

    timeseries_arr : np.ndarray
        array of shape (n_frames,n_atoms,3) holding coordinate or force information



    Returns
    -------

    np.ndarry of shape (n_frames,n_beads,3) after applying the CG map to every entry of timeseries_arr

    """
    assert (
        len(timeseries_arr.shape) == 3
    ), "Time series doesn't have shape (n_frames,n_atoms,3)"
    assert len(map_arr.shape) == 2, "Map doesn't have shape (n_beads,n_atoms)"
    assert (
        map_arr.shape[1] == timeseries_arr.shape[1]
    ), "`n_atoms` doesn't concide in the map and "
    if issubclass(type(map_arr), sparray):
        # scipy sparse arrays dont support the same broadcasting than numpy
        # we need to explicitly slice every frame and then stack them all
        return np.stack(
            [map_arr @ timeseries_arr[i, :, :] for i in range(timeseries_arr.shape[0])]
        )
    elif isinstance(map_arr, np.ndarray):
        # when using numpy non-sparse arrays, broadcasting over the frame dimension is supported
        return map_arr @ timeseries_arr
    else:
        raise ValueError(f"Map of type {type(map_arr)} is not supported")


def with_attrs(**func_attrs):
    """Set attributes in the decorated function, at definition time.
    Only accepts keyword arguments.
    """

    def attr_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        for attr, value in func_attrs.items():
            setattr(wrapper, attr, value)

        return wrapper

    return attr_decorator


def get_output_tag(tag_label: Union[List, str], placement: str = "before"):
    """
    Helper function for combining output tag labels neatly.
    Fixes issues of connecting/preceding '_' being included in some labels but not others.

    Parameters
    ----------
    tag_label : List, str
        Either a list of labels to include (ex: for datasets, delta force computation) or individual label item.
    placement : str
        Placement of tag in output name. One of: 'before', 'after'.
    """

    if isinstance(tag_label, str):
        if tag_label in [None, "", " "]:
            return ""
        else:
            return f"_{tag_label.strip('_')}"
    elif isinstance(tag_label, List):
        for l in tag_label:
            if l in [None, "", " "]:
                tag_label.remove(l)
        joined_label = "_".join([l.strip("_") for l in tag_label])
        if placement == "before":
            return f"{joined_label}_"
        elif placement == "after":
            return f"_{joined_label}"
        else:
            raise ValueError("Please specify placement from: 'before', 'after'.")


def map_cg_topology(
    atom_df: pd.DataFrame,
    cg_atoms: List[str],
    embedding_function: str,
    skip_residues: Optional[Union[List, str]] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    atom_df:
        Pandas DataFrame row from mdTraj topology.
    cg_atoms:
        List of atoms needed in CG mapping.
    embedding_function:
        Function that slices coodinates, if not provided will fail.
    special_typing:
        Optional dictionary of alternative atom properties to use in assigning types instead of atom names.
    skip_residues:
        Optional list of residues to skip when assigning CG atoms (can be used to skip caps for example);
        As of now, this skips all instances of a given residue.


    Returns
    -------
    New DataFrame columns indicating atom involvement in CG mapping and type assignment.

    Example
    -------
    First obtain a Pandas DataFrame object using the built-in MDTraj function:
    >>> top_df = aa_traj.topology.to_dataframe()[0]

    For a five-bead resolution mapping without including caps:
    >>> cg_atoms = ["N", "CA", "CB", "C", "O"]
    >>> embedding_function = embedding_fivebead
    >>> skip_residues = ["ACE", "NME"]

    Apply row-wise function:
    >>> top_df = top_df.apply(map_cg_topology, axis=1, cg_atoms, embedding_dict, skip_residues)
    """
    if isinstance(embedding_function, str):
        try:
            embedding_function = eval(embedding_function)
        except NameError:
            print("The specified embedding function has not been defined.")
            exit
    name, res = atom_df["name"], atom_df["resName"]
    if skip_residues != None and res in skip_residues:
        atom_df["mapped"] = False
        atom_df["type"] = "NA"

    else:
        if name in cg_atoms:
            atom_df["mapped"] = True
            atom_type = embedding_function(atom_df)
            atom_df["type"] = atom_type
        else:
            atom_df["mapped"] = False
            atom_df["type"] = "NA"

    return atom_df


def batch_matmul(map_matrix, X, batch_size):
    """
    Perform matrix multiplication in chunks.

    Parameters:
      map_matrix: Union[np.ndarray,sparray] of shape (N_CG_ats, N_FG_ats)
      X: np.ndarray of shape (M_frames, N_FG_ats, 3)
      batch_size: int, the number of rows (from the M dimension) to process at a time.

    Returns:
      result: np.ndarray of shape (M_frames, N_CG_ats, 3)
    """
    results = []
    M = X.shape[0]
    for i in range(0, M, batch_size):
        # Slice a batch along the M dimension
        X_batch = X[i : i + batch_size]  # shape: (batch, N, 3)
        # Perform matrix multiplication:
        # map_matrix (CG, FG) multiplied by each X_batch (FG, 3) gives (GC, 3) for each sample.
        # The broadcasting ensures the result is (batch, CG, 3)
        result_batch = cg_matmul(map_matrix, X_batch)
        results.append(result_batch)
    # Concatenate all chunks along the first axis (M dimension)
    return np.concatenate(results, axis=0)


def chunker(array, n_batches):
    """
    Chunks an input array into a specified number of batches.

    This function divides the input array into approximately equal-sized chunks.
    The last chunk may contain more elements if the array length is not perfectly
    divisible by the number of batches.

    Parameters:
    -----------
    array : np.ndarray or List
        The input array to be chunked.
    n_batches : int
        The number of batches to divide the array into. Must be a positive
        integer and less than or equal to the length of the array.

    Returns:
    --------
    batched_array: List
        A list of lists/arrays, where each inner list/array is a chunk of the original array.

    Examples:
    >>> chunker([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    >>> chunker([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4, 5]]

    >>> chunker([1, 2, 3, 4, 5], 5)
    [[1], [2], [3], [4], [5]]

    >>> chunker([1, 2, 3, 4, 5], 1)
    [[1, 2, 3, 4, 5]]
    """
    if n_batches == 1:
        return [array]
    assert n_batches <= len(
        array
    ), "n_batches needs to be smaller than the array to chunk"
    batched_array = []
    n_elts_per_batch = len(array) // n_batches
    for i in range(n_batches - 1):
        batched_array.append(array[i * n_elts_per_batch : (i + 1) * n_elts_per_batch])
    # last batch might be larger, it contains the rest of the elements in the array
    batched_array.append(array[(i + 1) * n_elts_per_batch :])
    return batched_array


def slice_coord_forces(
    coords,
    forces,
    cg_map,
    mapping: str = "slice_aggregate",
    force_stride: int = 100,
    batch_size: Optional[int] = None,
    atoms_batch_size: Optional[int] = None,
) -> Tuple:
    """
    Parameters
    ----------
    coords: [n_frames, n_atoms, 3]
        Numpy array of atomistic coordinates
    forces: [n_frames, n_atoms, 3]
        Numpy array of atomistic forces
    cg_map: [n_cg_atoms, n_atomistic_atoms]
        Linear map characterizing the atomistic to CG configurational map with shape.
        The code now supports also scipy.sparse arrays as cg_map types.
    mapping:
        Mapping scheme to be used,
        Can be either a string, then must be either 'slice_aggregate' or 'slice_optimize',
        Or can be directly a numpy array to use for projection
    force_stride:
        Striding to use for force projection results
    batch_size:
        Optional length of batch in which divide the AA mapping of coords and forces
        to CG ones
    atoms_batch_size:
        Optional batch size for dividing atoms in coordinates to estimate pairwise constraints

    Returns
    -------
    Coarse-grained coordinates and forces
    """
    # Original hard coded values
    n_frames = 100  # taking only first 100 frames gives same results in ~1/15th of time
    #                 (or even more in case "coords" is a DataIterable)
    threshold = 5e-3  # threshold for pairwise constraints

    config_map = aggforce.map.LinearMap(cg_map)
    config_map_matrix = config_map.standard_matrix

    # ensure the right coordinate/forces shape (useful when excluding solvent)
    n_sites = config_map.n_fg_sites
    if isinstance(coords, DataIterable):
        coords.atoms = coords.atoms[:n_sites]
    else:
        coords = coords[:, :n_sites, :]
    if isinstance(forces, DataIterable):
        forces.atoms = forces.atoms[:n_sites]
    else:
        forces = forces[:, :n_sites, :]
    if coords.shape[1] < n_sites:
        raise RuntimeError(f'not enough atoms in input coords (< {n_sites})')
    
    if atoms_batch_size is None or atoms_batch_size >= n_sites:
        # No batching: process all atoms at once
        print('Guessing pairwise constraints...')
        constraints = guess_pairwise_constraints(coords[:n_frames], threshold=threshold)

    else:
        # Batching mode
        batches = [
            (range(i, min(i + atoms_batch_size, n_sites)))
            for i in range(0, n_sites, atoms_batch_size)
        ]
        constraints = set()

        # extract the first n_frames as a numpy array
        # (irrelevant when coords is a numpy array,
        # crucial when it is a numpy iterator)
        xyz = coords[:n_frames]

        # Within-batch constraints
        for batch in tqdm(
            batches,
            desc='Guessing parwise constraints',
            unit='atom batches',
            position=0
        ):
            xyz_batch = xyz[:, batch, :]
            local_constraints = guess_pairwise_constraints(
                xyz_batch, threshold=threshold
            )
            global_constraints = {
                frozenset([batch[i] for i in pair]) for pair in local_constraints
            }
            constraints.update(global_constraints)

        # Cross-batch constraints
        # To significantly reduce computational cost, we assume residues are ordered in the structure.
        # Therefore, constraints are computed only between consecutive batches rather than all pairs of batches.
        # For even greater efficiency, this could be further limited to just the first and last (e.g., 30) atoms of each batch,
        # which scales approximately as O(1). However, computing all pairs between consecutive batches is generally still efficient.
        # This approach can also be extended to the case with no batching (for smaller molecules),
        # again assuming ordered residues, treating all molecules uniformly and eliminating the need for the atoms_batch_size parameter.
        for i in tqdm(range(len(batches) - 1),
                      desc='Cross-batch constraints',
                      unit='atom batches',
                      position=0
        ):
            b1 = batches[i]
            b2 = batches[i + 1]
            xyz1 = xyz[:, b1, :]
            xyz2 = xyz[:, b2, :]
            local_constraints = guess_pairwise_constraints(
                xyz1, cross_xyz=xyz2, threshold=threshold
            )
            # guess_pairwise_constraints returns ordered pairs (i, j) where i indexes into cross_xyz (b2) and j indexes into xyz (b1)
            global_constraints = {
                frozenset([b1[j], b2[i]]) for i, j in local_constraints
            }
            constraints.update(global_constraints)

    if isinstance(mapping, str):
        print('Calculating forces CG projection matrix...')
        if mapping == "slice_aggregate":
            method = constraint_aware_uni_map
            force_agg_results = project_forces(
                coords=coords[::force_stride],
                forces=forces[::force_stride],
                coord_map=config_map,
                constrained_inds=constraints,
                method=method,
            )
        elif mapping == "slice_optimize":
            method = qp_linear_map
            l2 = 1e3
            force_agg_results = project_forces(
                coords=coords[::force_stride],
                forces=forces[::force_stride],
                coord_map=config_map,
                constrained_inds=constraints,
                method=method,
                l2_regularization=l2,
            )
        else:
            raise RuntimeError(
                f"Force mapping {mapping} is neither 'slice_aggregate' nor 'slice_optimize'."
            )
        force_map_matrix = force_agg_results["tmap"].force_map.standard_matrix
    elif isinstance(mapping, np.ndarray):
        force_map_matrix = mapping
    else:
        raise RuntimeError(
            f"Force mapping {mapping} is neither a string nor a numpy array."
        )

    # convert to sparse arrays for better performance:
    config_map_matrix = csr_array(config_map_matrix)
    force_map_matrix = csr_array(force_map_matrix)

    print('Projecting coordinates and forces...')
    if isinstance(coords, DataIterable) or isinstance(forces, DataIterable):
        # if using DataIterable objects, it is more convenient to load/write
        # one frame at a time, since data are loaded to memory on-the-fly;
        # we actually need just one iterable, since "coords" trajectory also
        # contains the forces
        data = coords if isinstance(coords, DataIterable) else forces

        # initialize new (temporary) cg trajectory file
        trr = Path(data.trajectory.trajectory.filename)
        trr_fname = str(trr.with_name('.' + trr.stem + '_mapped.trr'))

        # get atoms
        atoms = data.atoms

        # initialize output cg universe
        n_cg_sites = config_map.n_cg_sites
        cg_universe = mda.Universe.empty(n_atoms=n_cg_sites, trajectory=True)
        cg_atoms = cg_universe.atoms
        cg_ts = cg_universe.trajectory.ts

        # write trajectory
        with TRRWriter(trr_fname, n_atoms=n_cg_sites) as writer:
            for aa_ts in tqdm(data.trajectory, position=0, unit='frames'):
                frame_coords = atoms.positions.reshape(-1, n_sites, 3)
                frame_forces = atoms.forces.reshape(-1, n_sites, 3) / 4.184
                # convert forces from kJ/A/mol to kCal/A/mol
                # reshape them such that matmul can handle them
                cg_ts.frame += 1
                cg_ts.time = aa_ts.time
                cg_ts.dimensions = aa_ts.dimensions
                cg_ts.positions = cg_matmul(config_map_matrix, frame_coords)[0]
                cg_ts.forces = cg_matmul(force_map_matrix, frame_forces)[0]
                writer.write(cg_atoms)
        
        # get mapped data iterables
        cg_trajectory = cg_universe.load_new(trr_fname).trajectory
        cg_coords = DataIterable(cg_trajectory, cg_atoms, 'positions')
        cg_forces = DataIterable(cg_trajectory, cg_atoms, 'forces', 1 / 4.184)

    elif batch_size is not None and coords.shape[0] > batch_size:
        cg_coords = batch_matmul(config_map_matrix, coords, batch_size=batch_size)
        cg_forces = batch_matmul(force_map_matrix, forces, batch_size=batch_size)
    else:
        cg_coords = cg_matmul(config_map_matrix, coords)
        cg_forces = cg_matmul(force_map_matrix, forces)

    return cg_coords, cg_forces, config_map_matrix, force_map_matrix


def filter_cis_frames(
    coords: Union[np.ndarray, DataIterable],
    forces: Union[np.ndarray, DataIterable],
    topology: md.Topology,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    filters out frames containing cis-omega angles

    Parameters
    ----------
    coords: [n_frames, n_atoms, 3]
        Non-filtered atomistic coordinates
    forces: [n_frames, n_atoms, 3]
        Non-filtered atomistic forces
    topology:
        mdtraj topology to load the coordinates with
    verbose:
        If True, will print a warning containing the number of discarded frames for this sample

    Returns
    -------
    Tuple of np.ndarray's for filtered coarse grained coordinates and forces
    """
    min_omega_atoms = set(["N", "CA", "C"])
    unique_atom_types = set([atom.name for atom in topology.atoms])
    if not min_omega_atoms.issubset(unique_atom_types):
        raise ValueError(
            "Provided pdb file must contain at least N, CA and C atoms for cis-omega filtering"
        )

    cis_omega_mask = np.zeros(coords.shape[0], dtype=bool)
    
    if isinstance(coords, DataIterable):
        # compute omega values frame-by-frame with DataIterable
        md_traj = None
        omega_values = []
        for i, xyz in enumerate(coords):
            if md_traj is None:
                md_traj = md.Trajectory(xyz, topology)
            else:
                md_traj.xyz = xyz
            omega_values.append(md.compute_omega(md_traj)[0])
    else:

        # compute omega values all at once
        md_traj = md.Trajectory(coords, topology)
        omega_idx, omega_values = md.compute_omega(md_traj)

    cis_omega_threshold = 1.0  # rad
    mask = np.all(np.abs(omega_values) > 1, axis=1)
    if not np.all(mask):
        warnings.warn(f"Discarding {len(mask) - np.sum(mask)} cis frames")
    if np.sum(mask) == 0:
        warnings.warn(f"This amounts to removing all frames for this molecule")

    return coords[mask], forces[mask]


def get_terminal_atoms(
    prior_builder: PriorBuilder,
    cg_dataframe: pd.DataFrame,
    N_term: Union[None, str] = None,
    C_term: Union[None, str] = None,
) -> Dict:
    """
    Parameters
    ----------
    prior_builder:

    cg_dataframe:
        Dataframe of CG topology (from MDTraj topology object).
    N_term: (Optional)
        Atom used in definition of N-terminus embedding.
    C_term: (Optional)
        Atom used in definition of C-terminus embedding.
    """
    chains = cg_dataframe.chainID.unique()
    # all atoms belonging to monopeptide chains will be removed from termini list
    monopeptide_atoms = []
    for chain in chains:
        residues = cg_dataframe.loc[cg_dataframe.chainID == chain].resSeq.unique()
        if len(residues) == 1:
            monopeptide_atoms.extend(
                cg_dataframe.loc[cg_dataframe.chainID == chain].index.to_list()
            )

    n_term_atoms = []
    c_term_atoms = []

    for chain in chains:
        chain_filter = cg_dataframe["chainID"] == chain
        first_res_chain, last_res_chain = (
            cg_dataframe[chain_filter]["resSeq"].min(),
            cg_dataframe[chain_filter]["resSeq"].max(),
        )
        n_term_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == first_res_chain) & chain_filter
            ].index.to_list()
        )
        c_term_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == last_res_chain) & chain_filter
            ].index.to_list()
        )

    prior_builder.n_term_atoms = [a for a in n_term_atoms if a not in monopeptide_atoms]
    prior_builder.c_term_atoms = [a for a in c_term_atoms if a not in monopeptide_atoms]

    N_term_name = "N" if N_term is None else N_term
    C_term_name = "C" if C_term is None else C_term

    n_term_name_atoms = []
    c_term_name_atoms = []
    for chain in chains:
        chain_filter = cg_dataframe["chainID"] == chain
        first_res_chain, last_res_chain = (
            cg_dataframe[chain_filter]["resSeq"].min(),
            cg_dataframe[chain_filter]["resSeq"].max(),
        )
        n_term_name_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == first_res_chain)
                & (cg_dataframe["name"] == N_term_name)
                & chain_filter
            ].index.to_list()
        )

        c_term_name_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == last_res_chain)
                & (cg_dataframe["name"] == C_term_name)
                & chain_filter
            ].index.to_list()
        )

    prior_builder.n_atoms = n_term_name_atoms
    prior_builder.c_atoms = c_term_name_atoms

    return prior_builder


def get_edges_and_orders(
    prior_builders: List[PriorBuilder],
    topology: md.Topology,
) -> List:
    """
    Parameters
    ----------
    prior_builders:
        List of PriorBuilder's to use for defining neighbour lists
    topology:
        MDTraj topology object from which atom groups defining each prior term will be created.
    cg_dataframe:
        Dataframe of CG topology (from MDTraj topology object).

    Returns
    -------
    List of edges, orders, and tag for each prior term specified in prior_dict.
    """
    all_edges_and_orders = []
    # process bond priors
    bond_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "bonds"
    ]
    all_bond_edges = []
    for prior_builder in bond_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
            all_bond_edges.extend(
                [p[2] for p in edges_and_orders if p[2].shape[1] != 0]
            )
        else:
            all_edges_and_orders.append(edges_and_orders)
            all_bond_edges.append(edges_and_orders[2])

    # process angle priors
    angle_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "angles"
    ]
    all_angle_edges = []
    for prior_builder in angle_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
            all_angle_edges.extend(
                [p[2] for p in edges_and_orders if p[2].shape[1] != 0]
            )
        else:
            all_edges_and_orders.append(edges_and_orders)
            all_angle_edges.append(edges_and_orders[2])

    # get nonbonded priors using bonded and angle edges
    if len(all_bond_edges) != 0:
        all_bond_edges = np.concatenate(all_bond_edges, axis=1)
    if len(all_angle_edges) != 0:
        all_angle_edges = np.concatenate(all_angle_edges, axis=1)

    nonbonded_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "non_bonded"
    ]
    for prior_builder in nonbonded_builders:
        edges_and_orders = prior_builder.build_nl(
            topology, bond_edges=all_bond_edges, angle_edges=all_angle_edges
        )
        # edges_and_orders = prior_dict[nbdict]["prior_function"](topology, all_bond_edges, all_angle_edges, **prior_dict[nbdict])
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)
    # process dihedral priors
    dihedral_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "dihedrals"
    ]
    for prior_builder in dihedral_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)

    return all_edges_and_orders


def split_bulk_termini(N_term, C_term, all_edges) -> Tuple:
    """
    Parameters
    ----------
    N_term:
        List of atom indices to be split as part of the N-terminal.
    C_term:
        List of atom indices to be split as part of the C-terminal.
    all_edges:
        All atom groups forming part of prior term.

    Returns
    -------
    Separated edges for bulk and terminal groups
    """
    n_term_idx = np.where(np.isin(all_edges.T, N_term))
    n_term_edges = all_edges[:, np.unique(n_term_idx[0])]

    c_term_idx = np.where(np.isin(all_edges.T, C_term))
    c_term_edges = all_edges[:, np.unique(c_term_idx[0])]

    term_edges = np.concatenate([n_term_edges, c_term_edges], axis=1)
    bulk_edges = np.array(
        [e for e in all_edges.T if not np.all(term_edges == e[:, None], axis=0).any()]
    ).T

    return n_term_edges, c_term_edges, bulk_edges


def get_dihedral_groups(
    top: md.Topology, atoms_needed: List[str], offset: List[int], tag: Optional[str]
) -> Dict:
    """
    Parameters
    ----------
    top:
        MDTraj topology object.
    atoms_needed: [4]
        Names of atoms forming dihedrals, should correspond to existing atom name in topology.
    offset: [4]
        Residue offset of each atom in atoms_needed from starting point.
    tag:
        Dihedral prior tag.

    Returns
    -------
    Dictionary of atom groups for each residue corresponding to dihedrals.

    Example
    -------
    To obtain all phi dihedral atom groups for a backbone-preserving resolution:
    >>> dihedral_dict = get_dihedral_groups(
    >>>     topology, atoms_needed=["C", "N", "CA", "C"], offset=[-1.,0.,0.,0.], tag="_phi"
    >>> )

    For a one-bead-per-residue mapping with only CA atoms preserved:
    >>> dihedral_dict = get_dihedral_groups(
    >>>     topology, atoms_needed=["CA", "CA", "CA", "CA"], offset=[-3.,-2.,-1.,0.]
    >>> )
    """
    res_per_chain = [[res for res in chain.residues] for chain in top.chains]
    atom_groups = {}
    for chain_idx, chain in enumerate(res_per_chain):
        for res in chain:
            res_idx = chain.index(res)
            if any(res_idx + ofs < 0 or res_idx + ofs >= len(chain) for ofs in offset):
                continue
            if any(atom not in [a.name for a in res.atoms] for atom in atoms_needed):
                continue
            label = f"{res.name}{tag}"
            if label not in atom_groups:
                atom_groups[label] = []
            dihedral = []
            for i, atom in enumerate(atoms_needed):
                atom_idx = top.select(
                    f"(chainid {chain_idx}) and (resid {res.index+offset[i]}) and (name {atom})"
                )
                dihedral.append(atom_idx)
            atom_groups[label].append(np.concatenate(dihedral))

    return atom_groups


def dataframe_to_mda_universe(df):
    """Convert dataframe extracted from mdtraj topology to a mda.Universe
    object, so to make it easier to do complex CG mappings."""
    names = df["name"].to_numpy()
    resnames = df["resName"].to_numpy()
    resids = df["resSeq"].to_numpy()
    segids = df["chainID"].to_numpy()
    n_atoms = len(df)
    if not n_atoms:
        return mda.Universe.empty(n_atoms=0,
                                  n_residues=0,
                                  atom_resindex=[],
                                  residue_segindex=[])
    residue_boundaries = [0] + (np.flatnonzero(
        np.diff(resids + segids * 1_000_000)) + 1).tolist() + [n_atoms]
    residue_n_atoms = np.diff(residue_boundaries)
    n_residues = len(residue_n_atoms)
    universe = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        n_segments=segids[-1] + 1,
        atom_resindex=np.repeat(range(n_residues), residue_n_atoms),
        residue_segindex=segids[residue_boundaries[:-1]],
    )
    universe.add_TopologyAttr("name", names)
    universe.add_TopologyAttr("resname", resnames[residue_boundaries[:-1]])
    universe.add_TopologyAttr("resid", resids[residue_boundaries[:-1]])
    universe.add_TopologyAttr("segid", np.arange(segids[-1] + 1).astype(str))
    return universe


def add_bonds(universe):
    """
    Add (fake) bonds to a MDAnalysis universe to be able to make chains
    whole when loading trajectories.
    """
    bonds = []
    # what is protein?
    is_protein = np.zeros(len(universe.atoms), dtype=bool)
    is_protein[universe.select_atoms('protein').indices] = True
    # step 1: proteins
    for segment in universe.segments:
        last_index = None
        for residue in segment.residues:
            indices = residue.atoms.indices
            first_index = indices[0]
            # with prior residue (only if protein)
            if is_protein[first_index] and last_index is not None:
                bonds.append([first_index, last_index])
            # within residue
            for index in indices[1:]:
                bonds.append([first_index, index])
            # update last index
            last_index = indices[-1]
    universe.add_bonds(bonds)
