"""
Common MPI tools (einsum, transpose) for coupled-cluster methods.

Author: Shuoxue Li <sli7@caltech.edu>
"""
from collections.abc import Iterable
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


class SegEinsumType(IntFlag):
    NONE = 0
    GENERAL = 1
    OUTER = 2
    MATMUL = 4
    BIDOT = 8
    NEWAXIS = 16


def get_vlocs(nvir):
    return [_task_location(nvir, task_id) for task_id in range(ntasks)]


def collect_array(arr: ArrayLike, seg_idx: int | None = None, debug: bool = False, sum_over: bool = True) -> ArrayLike:
    """
    For the segmented array, gather along the segmented indices;
    For the non-segmented array, return the sum for all the processes.
    """
    if seg_idx is None:
        # allreduce sends 
        return mpi.reduce(arr) if sum_over else arr
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
    if seg_idx is None: # non-segmented array, just broadcast it.
        arr = None
        if rank == 0:
            arr = arr_fn() if callable(arr_fn) else arr_fn
        arr = mpi.bcast(arr, root=0)
        return arr

    if rank == 0:
        arr = arr_fn() if callable(arr_fn) else arr_fn
        segs = [(slice(None), ) * seg_idx + (slice(*x), ) for x in vlocs]
        arr_segs = [arr[slc] for slc in segs]
        # if debug:
            # print(f"In get_mpi_array: {[x.shape for x in arr_segs]}")
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
    Outer product of the two segmented arrays: [x],[y]->[x]y or [x],[y]->x[y]
    
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
                 vlocss: list, seg_idxs_full: tuple, debug: bool = False) -> ArrayLike:
    """
    [x]y,[y]->[x] or [y],[x]y->[x]
    """
    vlocs1, vlocs2 = vlocss[:2]
    seg_idx12, seg_idx21 = seg_idxs_full[2:4]

    res = 0.
    # if rank == 0 and debug:
        # print(f"subscripts = {subscripts}, seg_idx12 = {seg_idx12}, seg_idx21 = {seg_idx21}")
        # print("arr1.shape = {}, arr2.shape = {}".format(arr1.shape, arr2.shape))

    if seg_idx21 is not None:  # xy,y->x
        rotate_fn = partial(_rotate_vir_block, vlocs=vlocs2)
        for _, arr2, p0, p1 in rotate_fn(arr2):
            _slc = (slice(None), ) * seg_idx21 + (slice(p0, p1), )
            # if rank == 0 and debug:
                # print("slc for arr1 = ", _slc)
            res += lib.einsum(subscripts, arr1[_slc], arr2)

    elif seg_idx12 is not None:   # y,xy->x
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
    [x]y,x[y]->...
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


def __newaxis_mpi(subscripts: str, arr1: ArrayLike, arr2: ArrayLike,
                  vlocs: list | None, seg_idxs_full: tuple, debug: bool = False):
    """
    [x],[x]y->[y] or [x]y,[x]->[y]
    """
    assert len(seg_idxs_full) > 6
    seg_idxf = seg_idxs_full[6]
    arr = lib.einsum(subscripts, arr1, arr2)
    arr = collect_array(arr, seg_idx=None)
    if vlocs is None:
        nvir_new = arr.shape[seg_idxf]
        vlocs = get_vlocs(nvir_new)
    return get_mpi_array(arr_fn=lambda: arr, vlocs=vlocs, seg_idx=seg_idxf, debug=debug)


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
    
    seg_type = SegEinsumType.NONE
    # get the type of segmented einsum
    if seg_str1 == seg_str2:
        if len(seg_idxs) == 3:
            if sub_f[seg_idxs[2]] != seg_str1:  # output has the different segmented index
                seg_type = SegEinsumType.NEWAXIS
            else:
                seg_type = SegEinsumType.GENERAL
        else:
            seg_type = SegEinsumType.GENERAL
    else:
        if sum((x is not None) for x in (seg_idx1, seg_idx2)) == 2:
            match sum((x is not None) for x in (seg_idx12, seg_idx21)):
                case 0: seg_type = SegEinsumType.OUTER
                case 1: seg_type = SegEinsumType.MATMUL
                case 2: seg_type = SegEinsumType.BIDOT
        else:
            seg_type = SegEinsumType.GENERAL

    assert seg_type != SegEinsumType.NONE, "The segmented einsum type cannot be NONE."


    subscripts_prtd = None
    if rank == 0 and debug:
        _fn_prtd = lambda s, i: s[:i] + "[" + s[i] + "]" + s[i+1:] if i is not None else s
        sub_i1_prtd = _fn_prtd(sub_i1, seg_idx1)
        sub_i2_prtd = _fn_prtd(sub_i2, seg_idx2)
        if len(seg_idxs) == 3:  # outer or newaxis case
            sub_f_prtd = _fn_prtd(sub_f, seg_idxs[2])
        else:
            if seg_idx1f is not None:
                sub_f_prtd = _fn_prtd(sub_f, seg_idx1f)
            elif seg_idx2f is not None:
                sub_f_prtd = _fn_prtd(sub_f, seg_idx2f)
            else:
                sub_f_prtd = sub_f
        subscripts_prtd = f"{sub_i1_prtd:<8}, {sub_i2_prtd:<8} -> {sub_f_prtd:<8}"

    segs = (seg_idx1, seg_idx2, seg_idx12, seg_idx21, seg_idx1f, seg_idx2f)
    if len(seg_idxs) >= 3:
        segs += tuple(seg_idxs[2:])

    if rank == 0 and debug:
        msg = f"   {subscripts_prtd:<32} Type:{seg_type.name:<14}"
        print(msg)

    # the segment index of the output array
    final_idx = [None, seg_idx1f, seg_idx2f, -1][sum((x is not None) << i for i, x in enumerate((seg_idx1f, seg_idx2f)))]

    # When the output array contains all the segmented indices,
    # the final index is either given by the user or the same as seg_idx1f.

    if final_idx == -1 and len(seg_idxs) == 2:
        final_idx = seg_idx1f
    if len(seg_idxs) == 3:
        final_idx = seg_idxs[2]

    # if rank == 0 and debug:
        # print(f"   Final segmented index of output array: {final_idx}\n")
    return seg_type, segs, final_idx


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
    seg_type, seg_idxs_full, final_idx = __get_segmented_einsum_type(subscripts, seg_idxs, debug=debug)
    arr1, arr2 = arrs

    if seg_type == SegEinsumType.GENERAL:
        res = lib.einsum(subscripts, arr1, arr2)
    else:
        assert seg_type != SegEinsumType.NONE, "The segmented einsum type cannot be NONE."

        if seg_type == SegEinsumType.NEWAXIS:
            vlocss = None

        res = {
            SegEinsumType.OUTER: __outer_mpi,
            SegEinsumType.MATMUL: __matmul_mpi,
            SegEinsumType.BIDOT: __bidot_mpi,
            SegEinsumType.NEWAXIS: __newaxis_mpi,}[seg_type](
             subscripts, arr1, arr2, vlocss, seg_idxs_full, debug=debug
         )
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
                 label: str = '', debug: bool = False):
        self.data = data
        self.seg_idx = seg_idx
        self.seg_spin = seg_spin
        self.label = label
        self.debug = debug

    # define negative operation
    def __neg__(self):
        return SegArray(-self.data, self.seg_idx, self.seg_spin, self.label)

    def __add__(self, other):
        _o = other if not isinstance(other, SegArray) else other.data
        if isinstance(other, SegArray):
            if self.seg_idx != other.seg_idx: # re-segment other to self.seg_idx
                if rank == 0 and self.debug:
                    print(f"In SegArray.__add__: merging different segmented indices\n"
                          f"self.seg_idx = {self.seg_idx}, other.seg_idx = {other.seg_idx}")
                has_seg_state = isinstance(self.seg_idx, int) + ((isinstance(other.seg_idx, int)) << 1)
                match has_seg_state:
                    case 3: # [self], [other]
                        _o = resegment(_o, seg_idx_old=other.seg_idx, seg_idx_new=self.seg_idx)
                    case 2: # self, [other]
                        _o = collect_array(_o, seg_idx=other.seg_idx)
                    case 1: # [self], other
                        _o = collect_array(_o, seg_idx=None)
                        vlocs = get_vlocs(_o.shape[self.seg_idx])
                        _o = get_mpi_array(lambda: _o, vlocs=vlocs, seg_idx=self.seg_idx, debug=False)
                    case _:
                        raise ValueError("Cannot identify the segmented state of the two arrays.")

        if isinstance(other, SegArray):
            other_label = other.label if other.label != "" else "unknown"
            print(f"Rank{rank}: {self.label} {self.data.shape} {other_label} {_o.shape}")

        # When the array does not have segment, just broadcast the added result
        if self.seg_idx is None:
            arr_fn = lambda: self.data + _o
            arr = get_mpi_array(arr_fn=arr_fn, vlocs=None, seg_idx=None, debug=self.debug)
        else:
            arr = self.data + _o

        return SegArray(arr, self.seg_idx, self.seg_spin, self.label, self.debug)

    def __mul__(self, other):
        _o = other.data if isinstance(other, SegArray) else other
        return SegArray(self.data * _o, self.seg_idx, self.seg_spin, self.label, self.debug)

    __rmul__ = __mul__

    # define subtraction operation
    def __sub__(self, other):
        return self + (-other)

    # define division operation
    def __truediv__(self, other):
        _o = other.data if isinstance(other, SegArray) else other
        return SegArray(self.data / _o, self.seg_idx, self.seg_spin, self.label, self.debug)

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
            return SegArray(data=self.data[key], seg_idx=idxs[0], label=self.label, debug=self.debug)

    def conj(self):
        return SegArray(self.data.conj(), self.seg_idx, self.seg_spin, self.label, self.debug)

    def transpose(self, *args):
        if len(args) == 1:
            tp = args[0]
        elif len(args) == 0:
            tp = tuple(range(self.data.ndim))[::-1]
        else:
            tp = args
        new_idx = tp.index(self.seg_idx)
        if rank == 0 and self.debug:
            print(f"In SegArray.transpose: old seg_idx = {self.seg_idx}, new seg_idx = {new_idx}, tp = {tp}")
        return SegArray(self.data.transpose(*args), new_idx, self.seg_spin, self.label, self.debug)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def collect(self):
        arr = collect_array(self.data, self.seg_idx, debug=self.debug)
        return SegArray(data=arr, seg_idx=None, seg_spin=None, label=self.label, debug=self.debug)
