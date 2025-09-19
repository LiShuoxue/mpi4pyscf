"""
mpi_parallel wrapper of the fundamental test functions.
Should be written in this separate file to avoid wrong module importing.
"""

import h5py
import numpy as np
from numpy.typing import ArrayLike

from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.ccsd import _task_location
from mpi4pyscf.cc import cc_tools as tools


rank = mpi.rank
comm = mpi.comm
ntasks = mpi.pool.size

SegArray = tools.SegArray

def _get_array(fname, key):
    with h5py.File(fname, 'r') as ftmp:
        arr = ftmp[key][:]
    return arr


def _get_segmented_array(fname: str, key: str, vlocs: list, seg_idx: int, debug: bool = False) -> ArrayLike:
    if rank == 0:
        arr = _get_array(fname, key)
        segs = [(slice(None), ) * seg_idx + (slice(*x), ) for x in vlocs]

        if debug:
            for i, seg in enumerate(segs):
                print(f"in __get_segmented_array: slices for rank {i} = {tools._tuple_of_slices_to_str(seg)}")
        arr_segs = [arr[seg] for seg in segs]

    else:
        arr, arr_segs = None, None

    # NOTE: Should use the default setting of 'data' arg to avoid unexpected reshape.
    arr_seg = mpi.scatter_new(arr_segs, root=0)
    arr, arr_segs = None, None
    return arr_seg


def _get_broadcast_array(fname: str, key: str) -> ArrayLike:
    print("rank =", rank, " is getting array", key, "from file", fname)
    if rank == 0:
        arr = _get_array(fname, key)
    else:
        arr = None
    arr = mpi.bcast(arr, root=0)
    return arr


def _debug_segmented_arr_info(arr: ArrayLike, arr_name: str, vlocs: list,
                              seg_idx: int, rk: int | None = None, sliced: bool = True):
    """"""
    rk = rank if rk is None else rk
    v0, v1 = vlocs[rk]
    slc = (slice(None), ) * seg_idx + (slice(v0, v1), )
    slc_str = tools._tuple_of_slices_to_str(slc)
    _arr = arr if sliced else arr[slc]

    msg = (
        f"Segmented array {arr_name}: rank = {rk}, vloc segment = [{v0}:{v1}]\n"
        f"    arr{slc_str} shape = {_arr.shape}\n"
        f"    arr{slc_str} norm = {np.linalg.norm(_arr)}\n"
        f"    First few elements of arr{slc_str}: {_arr.ravel()[:5]}\n"
    )
    print(msg)


@mpi.parallel_call
def test_basic_new(dev, fname: str):
    """
    Test fundamental functionalities including segmented einsum without SegArray class.
    """
    debug = True
    dim_A, dim_B, dim_C = 20, 24, 28
    vlocs_A, vlocs_B, vlocs_C = map(tools.get_vlocs, (dim_A, dim_B, dim_C))

    t_ACBB = _get_segmented_array(fname, 't_ACBB', vlocs=vlocs_C, seg_idx=1, debug=False)
    t_ABBB = _get_segmented_array(fname, 't_ABBB', vlocs=vlocs_B, seg_idx=1, debug=False)
    t_ABC = _get_segmented_array(fname, 't_ABC', vlocs=vlocs_B, seg_idx=1, debug=False)

    # Outer, idx0
    res_1 = tools.segmented_einsum_new(
        "acDE,aBDE->cB", arrs=(t_ACBB, t_ABBB),
        vlocss=(vlocs_C, vlocs_B), seg_idxs=(1, 1, 0), debug=debug
        )

    # Outer, idx1
    res_2 = tools.segmented_einsum_new(
        "acDE,aBDE->cB", arrs=(t_ACBB, t_ABBB),
        vlocss=(vlocs_C, vlocs_B), seg_idxs=(1, 1, 1), debug=debug
        )

    # Matmul: [x]y,[y]->x
    res_3 = tools.segmented_einsum_new(
        "aBc,acDE->aBDE", arrs=(t_ABC, t_ACBB),
        vlocss=(vlocs_B, vlocs_C), seg_idxs=(1, 1), debug=debug
        )

    # Matmul: [y],[x]y->x
    res_4 = tools.segmented_einsum_new(
        "ABc,ADBE->cDE", arrs=(t_ABC, t_ABBB),
        vlocss=(vlocs_B, vlocs_B), seg_idxs=(1, 1), debug=debug
        )

    # Common: x,x->...
    res_5 = tools.segmented_einsum_new(
        "ABc,ABDE->cD", arrs=(t_ABC, t_ABBB),
        vlocss=(None, None), seg_idxs=(1, 1), debug=debug
        )

    # Bidot: [x]y,x[y]->...
    res_6 = tools.segmented_einsum_new(
        'acBD,aBc->aD', arrs=(t_ACBB, t_ABC),
        vlocss=(vlocs_C, vlocs_B), seg_idxs=(1, 1), debug=debug
        )

    for r in (res_1, res_2, res_3, res_4, res_5, res_6):
        if debug and rank == 0:
            print(f"result shape = {r.shape}")

    res_1 = tools.collect_array(res_1, seg_idx=0)
    res_2 = tools.collect_array(res_2, seg_idx=1)
    res_3 = tools.collect_array(res_3, seg_idx=1)
    res_4 = tools.collect_array(res_4, seg_idx=1)
    res_5 = tools.collect_array(res_5, seg_idx=None)
    res_6 = tools.collect_array(res_6, seg_idx=None)
    return (res_1, res_2, res_3, res_4, res_5, res_6)


@mpi.parallel_call
def test_segarray_cls(dev, fname: str):
    """
    Test the SegArray class that wraps additional functionalities around a segmented array.
    """
    debug = True
    dim_A, dim_B, dim_C = 20, 24, 28
    vlocs_A, vlocs_B, vlocs_C = map(tools.get_vlocs, (dim_A, dim_B, dim_C))
    nvir_segA = vlocs_A[rank][1] - vlocs_A[rank][0]
    nvir_segB = vlocs_B[rank][1] - vlocs_B[rank][0]
    nvir_segC = vlocs_C[rank][1] - vlocs_C[rank][0]
    vlocs_dict = dict(A=vlocs_A, B=vlocs_B, C=vlocs_C)

    def _fn_getseg(key, seg_idx):
        seg_spin = (key.split("_")[1])[seg_idx]
        vlocs = vlocs_dict[seg_spin]
        arr = _get_segmented_array(fname, key, vlocs=vlocs, seg_idx=seg_idx, debug=False)
        return SegArray(arr, seg_idx=seg_idx, seg_spin=seg_spin)

    def _einsum(subscripts, arr1, arr2, out=None):
        vlocs_1 = vlocs_dict[arr1.seg_spin] if arr1.seg_spin else None
        vlocs_2 = vlocs_dict[arr2.seg_spin] if arr2.seg_spin else None
        seg_idxs = (arr1.seg_idx, arr2.seg_idx)
        final_spin = None
        if out is not None:
            seg_idxs += (out.seg_idx, )
            final_spin = out.seg_spin
        out_data, out_seg_idx = tools.segmented_einsum_new(
            subscripts=subscripts,
            arrs=(arr1.data, arr2.data),
            seg_idxs=seg_idxs,
            vlocss=(vlocs_1, vlocs_2),
            debug=debug,
            return_output_idx=True,
        )
        return SegArray(out_data, seg_idx=out_seg_idx, seg_spin=final_spin)

    if rank == 0:
        print("Start testing SegArray class ...")

    t_ACBB = _fn_getseg('t_ACBB', seg_idx=1)
    t_ABBB = _fn_getseg('t_ABBB', seg_idx=1)
    t_ABC  = _fn_getseg('t_ABC', seg_idx=1)

    res_1  = _einsum("acDE,aBDE->cB", t_ACBB, t_ABBB)                           # [c],[B]->[c]B

    res_2  = SegArray(np.zeros((dim_C, nvir_segB)), seg_idx=1, seg_spin='B')
    res_2 += _einsum("acDE,aBDE->cB", t_ACBB, t_ABBB)                           # [c],[B]->c[B]
    if rank == 0:
        print(f"res_2.shape = {res_2.shape}")

    res_3 = _einsum("aBc,acDE->aBDE", t_ABC, t_ACBB)                            # [B]c,[c]->[B]
    res_4 = _einsum("ABc,ADBE->cDE", t_ABC, t_ABBB)                             # [B],[D]B->[D]
    res_5 = _einsum("ABc,ABDE->cD", t_ABC, t_ABBB)                              # [B],[B]->...
    res_6 = _einsum("acBD,aBc->aD", t_ACBB, t_ABC)                              # [c]B,[B]c->...

    res_7  = SegArray(np.zeros((nvir_segB, dim_B)), seg_idx=0, seg_spin='B')
    res_7 += _einsum("acDE,acBE->BD", t_ACBB, t_ACBB)                # [c],[c]B->[B]

    res_8 = SegArray(np.zeros((dim_B, dim_B, nvir_segC)), seg_idx=2, seg_spin='C')  # [B]c,[B]->[c]
    res_8 += _einsum("aBc,aBDE->DEc", t_ABC, t_ABBB) * 2.

    _res_8T = res_8.set(debug=True).transpose(2, 0, 1)
    res_9 = res_4.set(debug=True) + _res_8T

    return tuple([x.collect().data for x in (res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8, res_9)])
