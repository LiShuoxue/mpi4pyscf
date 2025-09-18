import numpy as np
import h5py

from mpi4pyscf.tools import mpi
from pyscf import gto, lib
from libdmet.solver.scf import UIHF
from libdmet.solver.cc_solver import UICCSD
from mpi4pyscf.cc.uccsd import UICCSD as UICCSD_MPI
from mpi4pyscf.cc.test import _test_uccsd as testfuncs

rank = mpi.rank
comm = mpi.comm


def get_scf_solver():
    f = h5py.File("./data/ham_sz_sample.h5", 'r')
    h1e, eri, norb, ovlp = f['h1'][:], f['h2'][:], f['norb'][()], f['ovlp'][:]
    mol = gto.M().set(nao=norb, nelectron=norb, spin=0)
    mf = UIHF(mol).set(verbose=4, get_hcore=lambda *args: h1e,
                       get_ovlp = lambda *args: ovlp)
    mf._eri = eri
    mf.kernel()
    return mf


def get_vvvv(vvvv):
    assert vvvv.ndim == 2
    tmp = lib.unpack_tril(vvvv)
    nvir_pair1, nvir_2 = tmp.shape[:2]
    tmp = tmp.transpose(1, 2, 0).reshape(nvir_2 * nvir_2, nvir_pair1)
    tmp = lib.unpack_tril(tmp).transpose(1, 2, 0)
    nvir_1 = tmp.shape[0]
    return tmp.reshape(nvir_1, nvir_1, nvir_2, nvir_2)


def get_full_eris_mat(eris, key):
    nocca, noccb = eris.nocc
    nmoa, nmob = map(np.size, eris.mo_energy)
    nvira, nvirb = nmoa - nocca, nmob - noccb
    if key not in ('ovvv', 'OVVV', 'vvvv', 'VVVV', 'ovVV', 'OVvv', 'vvVV'):
        return getattr(eris, key)
    elif key in ('ovvv', 'OVVV', 'ovVV', 'OVvv'):
        slc = dict(v=slice(0, nvira), V=slice(0, nvirb))[key[-1]]
        return getattr(eris, f"get_{key}")(slc)
    else:
        return get_vvvv(getattr(eris, key))


def test_uiccsd_eris(mf: UIHF):
    cc_mpi = UICCSD_MPI(mf).set(verbose=5)
    res = testfuncs.test_uiccsd_eris(cc_mpi)
    eris_ref = None
    if rank == 0:
        eris_ref = UICCSD(mf).ao2mo()
        for seg_key in cc_mpi._eris._eri_keys:
            full_key = seg_key.replace('x', 'v').replace('X', 'V')
            arr_ref = get_full_eris_mat(eris_ref, full_key)
            diff = np.abs(res[full_key] - arr_ref).max()
            print (f"{full_key} (seg: {seg_key}) difference: {diff}")
            assert np.allclose(diff, 0.0)


def test_init_amps(mf: UIHF):
    cc_mpi = UICCSD_MPI(mf).set(verbose=4)
    res = testfuncs.test_init_amps(cc_mpi)
    if rank == 0:
        cc_ref = UICCSD(mf)
        cc_ref.ao2mo()
        E_ref, (t1a_ref, t1b_ref), (t2aa_ref, t2ab_ref, t2bb_ref) = cc_ref.init_amps()
        res_dict = dict(E=E_ref, t1a=t1a_ref, t1b=t1b_ref,
                         t2aa=t2aa_ref, t2ab=t2ab_ref, t2bb=t2bb_ref)
        for k in res:
            diff = np.abs(res[k] - res_dict[k]).max()
            print(f"Difference of {k}: {diff}")
            assert np.allclose(diff, 0.0)


if __name__ == "__main__":
    mf = get_scf_solver()
    test_uiccsd_eris(mf)
