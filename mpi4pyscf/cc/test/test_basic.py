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
    dim1, dim2, dim3, dim4 = 20, 24, 28, 25
    with h5py.File(fname, 'w') as ftmp:
        ftmp.create_dataset('t_AB', data=np.random.random(size=(dim1, dim2)))
        ftmp.create_dataset('t_BC', data=np.random.random(size=(dim2, dim3)))
        ftmp.create_dataset('t_ABC', data=np.random.random(size=(dim1, dim2, dim3)))
        ftmp.create_dataset('t_ABB', data=np.random.random(size=(dim1, dim2, dim2)))
        ftmp.create_dataset('t_ABBB', data=np.random.random(size=(dim1, dim2, dim2, dim2)))
        ftmp.create_dataset('t_ACBB', data=np.random.random(size=(dim1, dim3, dim2, dim2)))
        ftmp.create_dataset('t_ABDC', data=np.random.random(size=(dim1, dim2, dim4, dim3)))


def test_segd_einsum_type():
    from mpi4pyscf.cc.cc_tools import __get_segmented_einsum_type, SegEinsumType
    def _fn(subscripts, seg_idxs):
        return __get_segmented_einsum_type(subscripts, seg_idxs, debug=True)[0]

    assert _fn("nb,nemj->mbej", seg_idxs=(None, 1)) == SegEinsumType.GENERAL    # ...,[e]->[e]
    assert _fn("ac,bc->ab", seg_idxs=(1, 1)) == SegEinsumType.GENERAL           # [c],[c]->...
    assert _fn("nJfB,menf->mBeJ", seg_idxs=(2, 1)) == SegEinsumType.MATMUL      # [f],[e]f->[e]
    assert _fn("MNAF,MENF->AE", seg_idxs=(2, 1, 0)) == SegEinsumType.OUTER      # [A],[E]->[A]E
    assert _fn("acDE,aBDE->cB", seg_idxs=(1, 1, 1)) == SegEinsumType.OUTER      # [c],[B]->c[B]
    assert _fn("acBD,aBc->aD", seg_idxs=(1, 1)) == SegEinsumType.BIDOT          # [c]B,[B]c->...
    assert _fn("ac,cb->ab", seg_idxs=(1, 0, 0)) == SegEinsumType.NEWAXIS        # a[c],[c]->a
    assert _fn("ac,cb->ab", seg_idxs=(1, 0, 1)) == SegEinsumType.NEWAXIS        # [c],[c]b->b
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
        ref_7 = lib.einsum("acDE,acBE->BD", t_ACBB, t_ACBB)
        ref_8 = lib.einsum("aBc,aBDE->DEc", t_ABC, t_ABBB) * 2.
        ref_9 = ref_4 + ref_8.transpose(2, 0, 1)
        refs = (ref_1, ref_2, ref_3, ref_4, ref_5, ref_6, ref_7, ref_8, ref_9)

    ress = testfuncs.test_segarray_cls(None, fname)
    if rank == 0:
        print("All einsum data generated.\n")
        print("Difference between parallel and serial results:")
        for i, (res, ref) in enumerate(zip(ress, refs)):
            print(f"max diff of res_{i+1} = ", np.abs(res - ref).max())
            assert np.allclose(res, ref)


def test_dgemm():
    # NOTE dot test Correct.
    from pyscf.lib.numpy_helper import _dgemm
    m_tot, n_tot, k_tot = 200, 500, 300
    a = np.asarray(np.random.random(size=(m_tot, k_tot)), order='C')
    b = np.asarray(np.random.random(size=(k_tot, n_tot)), order='C')

    ref = a @ b
    blksize = 67

    c = np.zeros((m_tot, n_tot), order='C')

    for k0, k1 in lib.prange(0, k_tot, blksize):
        kc = k1 - k0
        _a = np.asarray(a[:, k0:k1], order='C')
        for n0, n1 in lib.prange(0, n_tot, blksize):
            nc = n1 - n0
            _b = np.asarray(b[k0:k1, n0:n1], order='C')
            _dgemm(trans_a='N', trans_b='N',
                    m=m_tot, n=nc, k=kc,
                    a=_a, b=_b, c=c,
                    alpha=1., beta=1.,
                    offseta=0, offsetb=0, offsetc=n0)

    print("max diff of dgemm = ", np.abs(c - ref).max() / np.abs(ref).max())


def test_scatter_seg():
    fname = './data/random_tensor.h5'
    testfuncs.test_scatter_seg(None, fname)


if __name__ == "__main__":
    # test_dgemm()
    # generate_test_data()
    test_scatter_seg()
    # if rank == 0:
        # test_segd_einsum_type()
        # test_segarr_cls()
    # test_basic_new()
