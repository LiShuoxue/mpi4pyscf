import numpy as np
import h5py
import pytest
from scipy import linalg as la

from mpi4pyscf.tools import mpi
from pyscf import gto, lib, scf, ao2mo
from pyscf.cc import uccsd as pyscf_uccsd
from pyscf.cc import uccsd_lambda as pyscf_uccsd_lambda

from libdmet.solver.scf import UIHF
from libdmet.solver.cc_solver import UICCSD
from mpi4pyscf.cc.uccsd import UICCSD as UICCSD_MPI
from mpi4pyscf.cc.test import _test_uccsd as testfuncs

rank = mpi.rank
comm = mpi.comm


def get_scf_solver_O2():
    mol = gto.Mole().set(
        atom=[
        [8 , (0., +2.000, 0.)],
        [8 , (0., -2.000, 0.)],],
        basis='cc-pvdz',
        spin=0,
    ).build()
    mf = scf.UHF(mol).run()
    moa, mob = mf.mo_coeff
    nmoa, nmob = moa.shape[1], mob.shape[1]
    np.random.seed(1919810)

    kappa_a = np.random.random((nmoa, nmoa))
    kappa_a = kappa_a - kappa_a.conj().T
    kappa_b = np.random.random((nmob, nmob))
    kappa_b = kappa_b - kappa_b.conj().T
    expK_a, expK_b = la.expm(kappa_a), la.expm(kappa_b)
    moa, mob = moa @ expK_a, mob @ expK_b

    eri_aa = ao2mo.restore(4, ao2mo.full(mf._eri, moa), nmoa)
    eri_bb = ao2mo.restore(4, ao2mo.full(mf._eri, mob), nmob)
    eri_ab = ao2mo.general(mf._eri, (moa, moa, mob, mob), compact=True)
    h1a = moa.conj().T @ mf.get_hcore() @ moa
    h1b = mob.conj().T @ mf.get_hcore() @ mob

    assert nmoa == nmob

    mf = UIHF(mol).set(verbose=4, get_hcore = lambda *args: np.array([h1a, h1b]),
                       get_ovlp = lambda *args: np.array([np.eye(nmoa), np.eye(nmob)]))
    mf._eri = (eri_aa, eri_bb, eri_ab)
    mf.kernel()
    return mf


def get_scf_solver_model():
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


def test_update_amps_checkpoint(mf: UIHF, checkpoint: int = 10):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    if rank == 0:
        print("Testing UCCSD update_amps with checkpointing ...")
    res = testfuncs.test_update_amps_checkpoint(cc_mpi, checkpoint=checkpoint)
    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()
        ref = testfuncs.update_amps_checkpoint_serial(cc_ref, t1_ref, t2_ref, eris, checkpoint)
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")
            # assert np.allclose(diff, 0.0)``


def test_update_amps(mf: UIHF):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    res = testfuncs.test_update_amps(cc_mpi)
    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()
        for _ in range(1):
            t1_ref, t2_ref = cc_ref.update_amps(t1_ref, t2_ref, eris=eris)
        ref = dict(t1a=t1_ref[0], t1b=t1_ref[1],
                         t2aa=t2_ref[0], t2ab=t2_ref[1], t2bb=t2_ref[2])
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")


def test_energy(mf: UIHF):
    max_cycle = 10
    cc_mpi = UICCSD_MPI(mf).set(verbose=5)
    res = testfuncs.test_energy(cc_mpi)
    if rank == 0:
        cc_ref = UICCSD(mf).set(verbose=5)
        eris = cc_ref.ao2mo()
        _, t1, t2 = cc_ref.init_amps()
        for _ in range(max_cycle):
            t1, t2 = cc_ref.update_amps(t1, t2, eris)
        E_ref = cc_ref.energy(t1=t1, t2=t2, eris=eris)
        diff = np.abs(res - E_ref)
        print(f"Difference of energy: {diff} / {np.abs(E_ref)} = {diff/np.abs(E_ref)}")
        assert np.allclose(diff, 0.0)


def test_lambda_intermediates_checkpoint(mf: UIHF, checkpoint: int = 10):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    res = testfuncs.test_lambda_intermediates_checkpoint(cc_mpi, checkpoint=checkpoint)
    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()
        ref = testfuncs.make_intermediates_checkpoint_serial(cc_ref, t1_ref, t2_ref,
                                                             eris=eris, checkpoint=checkpoint)
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")


def test_lambda_intermediates(mf: UIHF):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    res = testfuncs.test_lambda_intermediates(cc_mpi)
    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()
        imds_ref = pyscf_uccsd_lambda.make_intermediates(cc_ref, t1_ref, t2_ref, eris)
        for k in res:
            diff = np.abs(res[k] - getattr(imds_ref, k)).max()
            norm = np.linalg.norm(getattr(imds_ref, k))
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")


def test_update_lambda_checkpoint(mf: UIHF, checkpoint: int = 5):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    res, imds = testfuncs.test_update_lambda_checkpoint(cc_mpi, checkpoint=checkpoint)
    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()
        l1_ref, l2_ref = t1_ref, t2_ref
        imds_ref = pyscf_uccsd_lambda.make_intermediates(cc_ref, t1_ref, t2_ref, eris)
        ref = testfuncs.update_lambda_checkpoint_serial(cc_ref, t1=t1_ref, t2=t2_ref,
            l1=l1_ref, l2=l2_ref, eris=eris, imds=imds_ref, checkpoint=checkpoint)
        """
        for k in imds:
            diff = np.abs(imds[k] - getattr(imds_ref, k)).max()
            norm = np.linalg.norm(getattr(imds_ref, k))
            print(f"Difference of intermediate {k}: {diff} / {norm} = {diff/norm}")
        """
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of update_lambda {k}: {diff} / {norm} = {diff/norm}")


def test_update_lambda(mf: UIHF):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    res = testfuncs.test_update_lambda(cc_mpi)
    if rank == 0:
        cc_ref = UICCSD(mf)
        eris_ref = cc_ref.ao2mo()
        _, t10, t20 = cc_ref.init_amps()
        t1_ref, t2_ref = cc_ref.update_amps(t10, t20, eris=eris_ref)
        t1_ref, t2_ref = cc_ref.update_amps(t1_ref, t2_ref, eris=eris_ref)
        l1_ref, l2_ref = t10, t20
        imds_ref = pyscf_uccsd_lambda.make_intermediates(cc_ref, t1_ref, t2_ref, eris_ref)
        (l1a, l1b), (l2aa, l2ab, l2bb) = pyscf_uccsd_lambda.update_lambda(
            cc_ref, t1=t1_ref, t2=t2_ref, l1=l1_ref, l2=l2_ref, eris=eris_ref, imds=imds_ref
        )
        ref = dict(l1a=l1a, l1b=l1b, l2aa=l2aa, l2ab=l2ab, l2bb=l2bb)
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")
            assert np.allclose(diff, 0.0)


def test_ccsd_kernel(mf: UIHF):
    import cProfile, pstats
    profile_filename = f"test_uccsd.prof"

    pr = cProfile.Profile()
    pr.enable()
    cc_mpi = UICCSD_MPI(mf).set(verbose=5)
    cc_mpi.kernel()

    pr.disable()
    pr.dump_stats(profile_filename)

    if rank == 0:
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative').print_stats(60) # Print top 10 cumulative stats

    if rank == 0:
        cc_ref = UICCSD(mf).set(verbose=5)
        cc_ref.kernel()
        diff = np.abs(cc_mpi.e_corr - cc_ref.e_corr)
        print(f"Difference of CCSD energy: {diff} / {np.abs(cc_ref.e_corr)} = {diff/np.abs(cc_ref.e_corr)}")
        assert np.allclose(diff, 0.0)


if __name__ == "__main__":
    import sys

    task = sys.argv[1]
    if len(sys.argv) < 3: chkpt = None
    else: chkpt = int(sys.argv[2])

    mf = get_scf_solver_O2()

    if task == "eris": test_uiccsd_eris(mf)
    if task == "init_amps": test_init_amps(mf)
    if task == "tau": test_make_tau(mf)
    if task == "vvvv": test_add_vvvv(mf)
    if task == 'energy': test_energy(mf)
    if task == "kernel": test_ccsd_kernel(mf)

    if task == "amps":
        if chkpt is None: test_update_amps(mf)
        else: test_update_amps_checkpoint(mf, checkpoint=40)

    if task == "imds":
        if chkpt is None: test_lambda_intermediates(mf)
        else: test_lambda_intermediates_checkpoint(mf, checkpoint=chkpt)

    if task == "lambda":
        if chkpt is None: test_update_lambda(mf=mf)
        else: test_update_lambda_checkpoint(mf, checkpoint=chkpt)
