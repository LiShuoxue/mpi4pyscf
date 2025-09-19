import numpy as np
import h5py
import pytest

from mpi4pyscf.tools import mpi
from pyscf import gto, lib
from pyscf.cc import uccsd as pyscf_uccsd

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
    if rank == 0:
        print("Testing UCCSD integrals ...")
    cc_mpi = UICCSD_MPI(mf).set(verbose=5)
    res = testfuncs.test_uiccsd_eris(cc_mpi)
    eris_ref = None
    if rank == 0:
        eris_ref = UICCSD(mf).ao2mo()
        for seg_key in cc_mpi._eris._eri_keys:
            full_key = seg_key.replace('x', 'v').replace('X', 'V')
            arr_ref = get_full_eris_mat(eris_ref, full_key)
            diff = np.abs(res[full_key] - arr_ref).max()
            print (f"{full_key} (seg: {seg_key}) difference: {diff} / {np.linalg.norm(arr_ref)}")
            assert np.allclose(diff, 0.0)


def test_init_amps(mf: UIHF):
    if rank == 0:
        print("Testing UCCSD initial amplitudes ...")
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
            print(f"Difference of {k}: {diff} / {np.linalg.norm(res[k])}")
            assert np.allclose(diff, 0.0)


def test_make_tau(mf: UIHF):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    if rank == 0:
        print("Testing UCCSD tau amplitudes ...")
    res = testfuncs.test_make_tau(cc_mpi)
    if rank == 0:
        cc_ref = UICCSD(mf)
        _, t1_ref, t2_ref = cc_ref.init_amps()
        taus_ref = pyscf_uccsd.make_tau(t2_ref, t1_ref, t1_ref)
        keys = ('tauaa', 'taubb', 'tauab')
        ref = {k: taus_ref[i] for i, k in enumerate(keys)}
        print("Tau results:")
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")
            assert np.allclose(diff, 0.0)


def test_add_vvvv(mf: UIHF):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    if rank == 0:
        print("Testing UCCSD add_vvvv ...")
    res = testfuncs.test_add_vvvv(cc_mpi)
    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()
        oovvs_ref = pyscf_uccsd._add_vvvv(cc_ref, t1_ref, t2_ref, eris=eris)
        keys = ('oovv_aa', 'oovv_ab', 'oovv_bb')
        ref = {k: oovvs_ref[i] for i, k in enumerate(keys)}
        print("oovv results:")
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")
            assert np.allclose(diff, 0.0)


def test_update_amps(mf: UIHF):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    res = testfuncs.test_update_amps(cc_mpi)
    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()
        t1_ref, t2_ref = cc_ref.update_amps(t1_ref, t2_ref, eris=eris)
        ref = dict(t1a=t1_ref[0], t1b=t1_ref[1],
                         t2aa=t2_ref[0], t2ab=t2_ref[1], t2bb=t2_ref[2])
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")
            assert np.allclose(diff, 0.0), f"In test_update_amps, {k} failed."


if __name__ == "__main__":
    mf = get_scf_solver()

    test_uiccsd_eris(mf)
    test_init_amps(mf)
    test_make_tau(mf)
    test_add_vvvv(mf)
    test_update_amps(mf)
