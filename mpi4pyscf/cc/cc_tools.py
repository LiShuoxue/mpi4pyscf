"""
Common MPI tools (einsum, transpose) for coupled-cluster methods.

Author: Shuoxue Li <sli7@caltech.edu>
"""

from functools import partial
from enum import IntFlag
import numpy as np
from numpy.typing import ArrayLike
from pyscf import lib
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.ccsd import (_task_location, _rotate_vir_block, _cp)

comm = mpi.comm
rank = mpi.rank
ntasks = mpi.pool.size


KNOWN_SEG_TYPES = dict(
            general=(0b000000,      # ...,...->...
                     0b000001,      # x,x->...
                     0b010001,      # x,...->x; x,x->x
                     0b100010,      # ...,x->x
                    ),
            outer=(0b110011, ),     # x,y->xy
            matmul = (
                    0b011011,       # [x]y,[y]->x
                    0b100111,       # [x],x[y]->y
                ),
            bidot=(0b001111, )      # [x],[y]->xys
        )


def get_vlocs(nvir):
    return [_task_location(nvir, task_id) for task_id in range(ntasks)]


def collect_array(arr: ArrayLike, seg_idx: int | None = None, debug: bool = False, sum_over: bool = True) -> ArrayLike:
    """
    For the segmented array, gather along the segmented indices;
    For the non-segmented array, return the sum for all the processes.
    """
    if seg_idx is None:
        return mpi.allreduce(arr) if sum_over else arr
    else:
        ndim = arr.ndim
        tp = (seg_idx, ) + tuple(range(seg_idx)) + tuple(range(seg_idx+1, ndim))
        tp_inv = tuple(np.argsort(tp))
        if rank == 0 and debug:
            print(f"seg_idx = {seg_idx}, tp = {tp}, tp_inv = {tp_inv}")
        return mpi.gather_new(arr.transpose(tp)).transpose(tp_inv)


def get_mpi_array(arr_fn: callable, vlocs: list, seg_idx: int | None = None,
                  debug: bool = False) -> ArrayLike:
    """
    Get the array that is either non-segmented (seg_idx is None) or segmented by one index.
    """
    if seg_idx is None: # non-segmented array
        arr = None
        if rank == 0:
            arr = arr_fn()
        arr = mpi.bcast(arr, root=0)
        return arr

    if rank == 0:
        arr = arr_fn()
        segs = [(slice(None), ) * seg_idx + (slice(*x), ) for x in vlocs]
        arr_segs = [arr[slc] for slc in segs]
        if debug:
            print(f"In get_mpi_array: {[x.shape for x in arr_segs]}")
    else:
        arr, arr_segs = None, None
    arr_seg = mpi.scatter_new(arr_segs, root=0)
    arr, arr_segs = None, None
    return arr_seg


def resegment(arr: ArrayLike, seg_idx_old: int, seg_idx_new: int, vlocs: list | None = None, debug: bool = False):
    # A rough version: collect then re-segment
    gathered_arr = collect_array(arr, seg_idx=seg_idx_old, debug=debug)
    if vlocs is None:
        nvir_new = gathered_arr.shape[seg_idx_new]
        vlocs = get_vlocs(nvir_new)
    if rank == 0 and debug:
        print(f"In __resegment: Before re-segmentation: gathered_arr.shape = {gathered_arr.shape}")
    return get_mpi_array(lambda: gathered_arr, vlocs=vlocs, seg_idx=seg_idx_new)


def __outer_mpi(subscripts: str, arr1: ArrayLike, arr2: ArrayLike,
                vlocss: list, seg_idxs_full: tuple, debug: bool = False) -> ArrayLike:
    """
    Outer product of the two segmented arrays.
    
    seg_idxs: should have 3 elements, indicating the segmented index of arr1, arr2, and the output array.s
    vlocss: task locations of the two segmented arrays arr1 and arr2.
    """
    sub_i, sub_f = subscripts.split("->")
    sub_i1, sub_i2 = sub_i.split(",")
    vlocs1, vlocs2 = vlocss

    seg_idx1, seg_idx2 = seg_idxs_full[:2]

    if len(seg_idxs_full) > 6:
        seg_idxf = seg_idxs_full[6]
    else:
        seg_idxf = seg_idxs_full[4]

    if sub_f[seg_idxf] == sub_i1[seg_idx1]:
        vlocs, Y_idx_final = vlocs2, sub_f.index(sub_i2[seg_idx2])
    elif sub_f[seg_idxf] == sub_i2[seg_idx2]:
        vlocs, Y_idx_final = vlocs1, sub_f.index(sub_i1[seg_idx1])
    else:
        raise ValueError("The segmented index of output array should be the same as one of the input arrays.")

    rotate_fn = partial(_rotate_vir_block, vlocs=vlocs)
    shapes = [arr1.shape[sub_i1.index(s)] if s in sub_i1 else arr2.shape[sub_i2.index(s)] for s in sub_f]
    shapes[Y_idx_final] = vlocs[-1][1]
    res = np.zeros(tuple(shapes), dtype=arr1.dtype)

    if sub_f[seg_idxf] == sub_i1[seg_idx1]:    # x as final segd idx
        for _, arr2, p0, p1 in rotate_fn(arr2):
            _r = lib.einsum(subscripts, arr1, arr2) # xy index
            _slc = (slice(None), ) * Y_idx_final + (slice(p0, p1), )
            res[_slc] += _r
    elif sub_f[seg_idxf] == sub_i2[seg_idx2]:
        for _, arr1, p0, p1 in rotate_fn(arr1):
            _r = lib.einsum(subscripts, arr1, arr2)
            _slc = (slice(None), ) * Y_idx_final + (slice(p0, p1), )
            res[_slc] += _r
    return res


def __matmul_mpi(subscripts: str, arr1: ArrayLike, arr2: ArrayLike,
                 vlocss: list, seg_idxs_full: tuple, has_idx: int, debug: bool = False) -> ArrayLike:
    """
    [x]y,[y]->x or [y],[x]y->x
    """
    assert has_idx in (0b011011, 0b100111)
    vlocs1, vlocs2 = vlocss[:2]
    seg_idx12, seg_idx21 = seg_idxs_full[2:4]

    res = 0.
    # if rank == 0 and debug:
        # print(f"subscripts = {subscripts}, seg_idx12 = {seg_idx12}, seg_idx21 = {seg_idx21}")
        # print("arr1.shape = {}, arr2.shape = {}".format(arr1.shape, arr2.shape))

    if has_idx == 0b011011:  # xy,y->x
        rotate_fn = partial(_rotate_vir_block, vlocs=vlocs2)
        for _, arr2, p0, p1 in rotate_fn(arr2):
            _slc = (slice(None), ) * seg_idx21 + (slice(p0, p1), )
            # if rank == 0 and debug:
                # print("slc for arr1 = ", _slc)
            res += lib.einsum(subscripts, arr1[_slc], arr2)

    elif has_idx == 0b100111:   # y,xy->x
        rotate_fn = partial(_rotate_vir_block, vlocs=vlocs1)
        for _, arr1, p0, p1 in rotate_fn(arr1):
            _slc = (slice(None), ) * seg_idx12 + (slice(p0, p1), )
            # if rank == 0 and debug:
                # print("slc for arr2 = ", _slc)
            res += lib.einsum(subscripts, arr1, arr2[_slc])

    return res


def __bidot_mpi(subscripts: str, arr1: ArrayLike, arr2: ArrayLike,
                vlocss: list, seg_idxs_full: tuple, debug: bool = False) -> ArrayLike:
    """
    xy,xy->...
    Return the array that should be reduced along x axis.
    """
    vlocs1, vlocs2 = vlocss
    seg_idx12, seg_idx21 = seg_idxs_full[2:4]
    rotate_fn = partial(_rotate_vir_block, vlocs=vlocs2)
    v0, v1 = vlocs1[rank]

    res = 0.
    _slc_12 = (slice(None), ) * seg_idx12 + (slice(v0, v1), )
    for _, arr2, p0, p1 in rotate_fn(arr2):
        _slc_21 = (slice(None), ) * seg_idx21 + (slice(p0, p1), )
        res += lib.einsum(subscripts, arr1[_slc_21], arr2[_slc_12])
    return res


# def __adjacent_mpi(subsripts: str)


def __get_segmented_einsum_type(subscripts: str, seg_idxs: tuple, debug: bool = False) -> int:
    """
    Return:
     has_idx: shows the einsum type, an integer where for each bit: (
        seg on arr1; seg on arr2;
        idx of segd arr1 in arr2; idx of segd arr2 in arr1;
        idx of segd arr1 in output; idx of segd arr2 in output.
        )
     segs: the tuple of the 6 segmented indices.
    """
    sub_i, sub_f = subscripts.split("->")
    sub_i1, sub_i2 = sub_i.split(",")
    seg_idx1, seg_idx2 = seg_idxs[:2]
    seg_str1 = None if seg_idx1 is None else sub_i1[seg_idx1]
    seg_str2 = None if seg_idx2 is None else sub_i2[seg_idx2]

    # segmented index of seg_idx1 in array 2.
    seg_idx12 = None if (seg_str1 is None or not seg_str1 in sub_i2) else sub_i2.index(seg_str1)
    # segmented index of seg_idx2 in array 1.
    seg_idx21 = None if (seg_str2 is None or not seg_str2 in sub_i1) else sub_i1.index(seg_str2)
    seg_idx1f = None if (seg_str1 is None or not seg_str1 in sub_f) else sub_f.index(seg_str1)
    seg_idx2f = None if (seg_str2 is None or not seg_str2 in sub_f) else sub_f.index(seg_str2)

    _repr = lambda s: "N" if s is None else s

    subscripts_prtd = None
    if rank == 0 and debug:
        _fn_prtd = lambda s, i: s[:i] + "[" + s[i] + "]" + s[i+1:] if i is not None else s
        sub_i1_prtd = _fn_prtd(sub_i1, seg_idx1)
        sub_i2_prtd = _fn_prtd(sub_i2, seg_idx2)
        if len(seg_idxs) == 3:  # outer
            sub_f_prtd = _fn_prtd(sub_f, seg_idxs[2])
        else:
            if seg_idx1f is not None:
                sub_f_prtd = _fn_prtd(sub_f, seg_idx1f)
            elif seg_idx2f is not None:
                sub_f_prtd = _fn_prtd(sub_f, seg_idx2f)
            else:
                sub_f_prtd = sub_f
        subscripts_prtd = f"{sub_i1_prtd}, {sub_i2_prtd} -> {sub_f_prtd}"

    # If the two segmented indices are the same, then the has_idx label should be changeds to avoid conflicts.
    if seg_str1 == seg_str2:    # x,x->...
        seg_idx12, seg_idx21, seg_idx2, seg_idx2f = None, None, None, None

    segs = (seg_idx1, seg_idx2, seg_idx12, seg_idx21, seg_idx1f, seg_idx2f)
    has_idx = sum((x is not None) << i for i, x in enumerate(segs))

    if rank == 0 and debug:
        for k, v in KNOWN_SEG_TYPES.items():
            if has_idx in v:
                print(f"   {subscripts_prtd:<28} {k:<14} Idxs 1:{_repr(seg_idx1)} 2:{_repr(seg_idx2)} "
              f"1in2:{_repr(seg_idx12)} 2in1:{_repr(seg_idx21)} 1inf:{_repr(seg_idx1f)} 2inf:{_repr(seg_idx2f)}")
                break

    if len(seg_idxs) >= 3:
        segs += tuple(seg_idxs[2:])

    # the segment index of the output array
    final_idx = [None, seg_idx1f, seg_idx2f, -1][has_idx >> 4]

    # When the output array contains all the segmented indices,
    # the final index is either given by the user or the same as seg_idx1f.
    if final_idx == -1:
        # assert len(seg_idxs) == 3, "When the output array is segmented by both indices, the seg_idx should be given."
        if len(seg_idxs) == 3:
            final_idx = seg_idxs[2]
        else:
            final_idx = seg_idx1f

    return has_idx, segs, final_idx


def segmented_einsum_new(subscripts: str, arrs: tuple[ArrayLike, ArrayLike],
                         seg_idxs: tuple, vlocss: tuple | None = None, debug: bool = False,
                         return_output_idx: bool = False) -> ArrayLike:
    """
    Given the segmented indices of the two input arrays, perform the segmented einsum with less restriction of the input.
    Example:
        wOvVo += 0.5*lib.einsum('jNbF,MENF->MbEj', t2ab, OVOV)
            t2ab -> t2_oOxV, seg_idx = 2 (x)
            OVOV -> OXOV, seg_idx = 1 (X)
            wOvVo -> wOvXo, seg_idx = 2 (X)
            Output arrays should be jNyF,MXNF->MyXj, outer, then merge the index y and return X for each proc.
    """
    has_idx, seg_idxs_full, final_idx = __get_segmented_einsum_type(subscripts, seg_idxs, debug=debug)
    arr1, arr2 = arrs

    seg_type = 'nothing'
    for k, v in KNOWN_SEG_TYPES.items():
        if has_idx in v:
            seg_type = k
            break

    if seg_type == "general":
        res = lib.einsum(subscripts, arr1, arr2)
    elif seg_type == "outer":
        res = __outer_mpi(subscripts, arr1, arr2, vlocss, seg_idxs_full, debug=debug)
    elif seg_type == 'matmul':
        res = __matmul_mpi(subscripts, arr1, arr2, vlocss, seg_idxs_full, has_idx, debug=debug)
    elif seg_type == 'bidot':
        res = __bidot_mpi(subscripts, arr1, arr2, vlocss, seg_idxs_full, debug=debug)
    else:
        raise NotImplementedError("segmented einsum type %s is not implemented" % seg_type)

    if return_output_idx:
        return res, final_idx
    else:
        return res


def _tuple_of_slices_to_str(slices: tuple) -> str:
    def _slice_to_str(s: slice) -> str:
        """Convert a slice object to a string like Python slice notation."""
        if s == slice(None):  # full slice
            return ":"
        parts = []
        parts.append("" if s.start is None else str(s.start))
        parts.append("" if s.stop is None else str(s.stop))
        if s.step is not None:
            parts.append(str(s.step))
        return ":".join(parts)
    return "[" + ", ".join(_slice_to_str(s) for s in slices) + "]"


def segmented_transpose(arr: ArrayLike, seg_idx: int, tp_idx: int, coeff_0: float, coeff: float,
                          vlocs: list, debug: bool = False) -> ArrayLike:
    """
    Support the transpose of two indices.
    M = M * coeff_0 + M.transpose(tp) * coeff
    """
    tmp = _cp(arr) * coeff_0
    vloc0, vloc1 = vlocs[rank]
    slc_seg = slice(vloc0, vloc1)
    slc_seg_rel = slice(0, vloc1 - vloc0)
    min_idx, max_idx = min(tp_idx, seg_idx), max(tp_idx, seg_idx)
    tp = tuple(range(min_idx)) + (max_idx, ) + tuple(range(min_idx+1, max_idx)) + (min_idx, ) + tuple(range(max_idx+1, arr.ndim))

    rotate_fn = partial(_rotate_vir_block, vlocs=vlocs)

    for task_id, arr, p0, p1 in rotate_fn(arr):
        slc_tp = slice(p0, p1)
        slc_tp_rel = slice(0, p1 - p0)
        if seg_idx < tp_idx:
            slc_0 = (slice(None), ) * min_idx + (slc_seg_rel, ) + (slice(None), ) * (max_idx - min_idx - 1) + (slc_tp, )
            slc_1 = (slice(None), ) * min_idx + (slc_seg, ) + (slice(None), ) * (max_idx - min_idx - 1) + (slc_tp_rel, )
        else:
            slc_0 = (slice(None), ) * min_idx + (slc_tp, ) + (slice(None), ) * (max_idx - min_idx - 1) + (slc_seg_rel, )
            slc_1 = (slice(None), ) * min_idx + (slc_tp_rel, ) + (slice(None), ) * (max_idx - min_idx - 1) + (slc_seg, )

        if debug:
            msg = (
                f"rank = {rank}, task_id = {task_id}, p0 = {p0}, p1 = {p1}\n"
                f"slc_seg = {slc_seg}, slc_tp = {slc_tp}\n"
                f"slc_0 = {_tuple_of_slices_to_str(slc_0)}, slc_1 = {_tuple_of_slices_to_str(slc_1)}\n"
            )
            print(msg)

        tmp[slc_0] += arr.transpose(tp)[slc_1] * coeff
    return tmp


class SegArray(lib.StreamObject):
    """
    A wrapped class of the segmented array.

    Attributes:
        data: the local array on each MPI process.
        seg_idx: the segmented index.
        seg_spin: any type, the spin (or other kind of label) of the segmented index, default None.
        label: a string label for the array.
    """
    def __init__(self, data: ArrayLike,
                 seg_idx: int | None = None, seg_spin = None,
                 label: str = ''):
        self.data = data
        self.seg_idx = seg_idx
        self.seg_spin = seg_spin
        self.label = label

    # define negative operation
    def __neg__(self):
        return SegArray(-self.data, self.seg_idx, self.seg_spin, self.label)

    def __add__(self, other):
        _o = other if not isinstance(other, SegArray) else other.data
        if isinstance(other, SegArray):
            if self.seg_idx != other.seg_idx:
                # re-segment other to self.seg_idx
                assert isinstance(self.seg_idx, int) and isinstance(other.seg_idx, int)
                _o = resegment(_o, seg_idx_old=other.seg_idx, seg_idx_new=self.seg_idx)
        return SegArray(self.data + _o, self.seg_idx, self.seg_spin, self.label)

    def __mul__(self, other):
        _o = other.data if isinstance(other, SegArray) else other
        return SegArray(self.data * _o, self.seg_idx, self.seg_spin, self.label)

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + (-other)

    def __repr__(self):
        return (f"SegArray({self.label}) <seg_idx={self.seg_idx}, seg_spin={self.seg_spin}>: \n"
                f"{self.data.__repr__()}")

    def __getitem__(self, key):
        """Define the indexing operation"""
        if isinstance(key, slice | int):
            key = (key, )
        assert isinstance(key, tuple)
        idxs = [i for i, kk in enumerate(key) if kk != slice(None)]
        if len(idxs) == 0:
            return self
        else:
            assert self.seg_idx is None, "Indexing should be used for non-segmented array."
            return SegArray(data=self.data[key], seg_idx=idxs[0], label=self.label)

    def conj(self):
        return SegArray(self.data.conj(), self.seg_idx, self.seg_spin, self.label)

    def transpose(self, *args):
        if len(args) == 1:
            tp = args[0]
        elif len(args) == 0:
            tp = tuple(range(self.data.ndim))[::-1]
        else:
            tp = args
        new_idx = tp.index(self.seg_idx)
        return SegArray(self.data.transpose(*args), new_idx, self.seg_spin, self.label)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def collect(self):
        return SegArray(collect_array(self.data, self.seg_idx),
                        seg_idx=None, seg_spin=None, label=self.label)
