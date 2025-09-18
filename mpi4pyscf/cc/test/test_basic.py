import numpy as np
import h5py
from functools import partial
from pyscf import lib
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.test import _test_basic as testfuncs

rank = mpi.rank
comm = mpi.comm
ntasks = mpi.pool.size


def generate_test_data():
    fname = "./data/random_tensor.h5"
    dim1, dim2, dim3 = 20, 24, 28
    with h5py.File(fname, 'w') as ftmp:
        ftmp.create_dataset('t_AB', data=np.random.random(size=(dim1, dim2)))
        ftmp.create_dataset('t_BC', data=np.random.random(size=(dim2, dim3)))
        ftmp.create_dataset('t_ABC', data=np.random.random(size=(dim1, dim2, dim3)))
        ftmp.create_dataset('t_ABB', data=np.random.random(size=(dim1, dim2, dim2)))
        ftmp.create_dataset('t_ABBB', data=np.random.random(size=(dim1, dim2, dim2, dim2)))
        ftmp.create_dataset('t_ACBB', data=np.random.random(size=(dim1, dim3, dim2, dim2)))


def test_segd_einsum_type():
    from mpi4pyscf.cc.cc_tools import __get_segmented_einsum_type
    def _fn(subscripts, seg_idxs):
        return __get_segmented_einsum_type(subscripts, seg_idxs, debug=True)[0]

    assert _fn("nb,nemj->mbej", seg_idxs=(None, 1)) == 0b100010  # t1, OXOO: general
    assert _fn("ac,bc->ab", seg_idxs=(1, 1)) == 0b000001  # x,x->x
    assert _fn("nJfB,menf->mBeJ", seg_idxs=(2, 1)) == 0b100111    # t2_oOxV, oxov: matmul
    assert _fn("MNAF,MENF->AE", seg_idxs=(2, 1, 0)) == 0b110011       # til_OOXV, OXOV: outer
    assert _fn("acDE,aBDE->cB", seg_idxs=(1, 1, 1)) == 0b110011   # outer
    assert _fn("acBD,aBc->aD", seg_idxs=(1, 1)) == 0b001111    # bidot
    print("All segmented einsum type tests passed!\n")


def test_segarr_cls():
    SegArray = testfuncs.SegArray
    arr = SegArray(np.random.random(size=(7, 7, 7)), seg_idx=None, seg_spin=None, label='t_ABC')
    print(arr[2:4], "\n", arr[:, 4:6])

def test_basic_new():
    fname = "./data/random_tensor.h5"

    if rank == 0:
        t_ABC, t_ACBB, t_ABBB = map(lambda x: testfuncs._get_array(fname, x), ('t_ABC', 't_ACBB', 't_ABBB'))
        ref_1 = lib.einsum("acDE,aBDE->cB", t_ACBB, t_ABBB)
        ref_2 = ref_1
        ref_3 = lib.einsum("aBc,acDE->aBDE", t_ABC, t_ACBB)
        ref_4 = lib.einsum("ABc,ADBE->cDE", t_ABC, t_ABBB)
        ref_5 = lib.einsum("ABc,ABDE->cD", t_ABC, t_ABBB)
        ref_6 = lib.einsum("acBD,aBc->aD", t_ACBB, t_ABC)
        refs = (ref_1, ref_2, ref_3, ref_4, ref_5, ref_6)

    ress = testfuncs.test_segarray_cls(None, fname)

    if rank == 0:
        print("All einsum data generated.\n")
        print("Difference between parallel and serial results:")
        for i, (res, ref) in enumerate(zip(ress, refs)):
            print(f"max diff of result {i} = ", np.abs(res - ref).max())
            assert np.allclose(res, ref)


if __name__ == "__main__":
    if rank == 0:
        test_segd_einsum_type()
        # test_segarr_cls()
    test_basic_new()
