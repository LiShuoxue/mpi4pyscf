import numpy as np
import h5py
import os
from scipy import linalg as la

from mpi4pyscf.tools import mpi
from pyscf import gto, lib, scf, ao2mo
from pyscf.cc import uccsd as pyscf_uccsd
from pyscf.cc import uccsd_lambda as pyscf_uccsd_lambda

from libdmet.solver.scf import UIHF
from libdmet.solver.cc_solver import UICCSD, UICCSD_KRYLOV
from mpi4pyscf.cc.uccsd import UICCSD as UICCSD_MPI
from mpi4pyscf.cc.uccsd_krylov import UICCSD_KRYLOV as UICCSD_KRYLOV_MPI
from mpi4pyscf.cc.test import _test_uccsd as testfuncs

import cProfile

rank = mpi.rank
comm = mpi.comm


def get_scf_solver_mol(mol, chkfile: str = None):
    print(f"Molecule Information: nao = {mol.nao}, nelec = {mol.nelec}")
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

    mf = UIHF(mol).set(verbose=4, get_hcore=lambda *args: np.array([h1a, h1b]),
                       get_ovlp=lambda *args: np.array([np.eye(nmoa), np.eye(nmob)]),
                       chkfile=chkfile)
    mf._eri = (eri_aa, eri_bb, eri_ab)

    if os.path.isfile(str(chkfile)):
        print("Loading SCF from checkpoint file:", chkfile)
        mf.__dict__.update(scf.chkfile.load_scf(chkfile)[1])
    else:
        print("Running SCF ...")
        mf.kernel()
    return mf


def get_scf_solver_model(fname: str):
    # fname = "./data/ham_sz_sample.h5"
    f = h5py.File(fname, 'r')
    h1e, eri, norb, ovlp = f['h1'][:], f['h2'][:], f['norb'][()], f['ovlp'][:]
    mol = gto.M().set(nao=norb, nelectron=norb, spin=0)
    mf = UIHF(mol).set(verbose=4, get_hcore=lambda *args: h1e,
                       get_ovlp=lambda *args: ovlp)
    mf._eri = eri
    mf.kernel()
    return mf


def get_scf_solver_O2():
    mol = gto.Mole().set(
        atom=[
        [8 , (0., +2.000, 0.)],
        [8 , (0., -2.000, 0.)],],
        basis='cc-pvdz',
        spin=0,
    ).build()
    return get_scf_solver_mol(mol)


def get_scf_solver_S2():
    mol = gto.Mole().set(
        atom=[
        [16 , (0., +2.700, 0.)],
        [16 , (0., -2.700, 0.)],],
        basis='cc-pvtz',
        spin=0,
    ).build()
    return get_scf_solver_mol(mol)


# def get_scf_solver_CCO():
    # f = h5py.File("/resnick/scratch/syuan/202509-svp/cco/k222-pyscfmkl/x0.00/UDMET-xcsc/ham_imp_0.h5", 'r')
    # h1e, eri, norb, ovlp = f['h1'][:], f['h2'][:], f['norb'][()], f['ovlp'][:]
    # mol = gto.M().set(nao=norb, nelectron=norb, spin=0)
    # mf = UIHF(mol).set(verbose=4, get_hcore=lambda *args: h1e,
                       # get_ovlp=lambda *args: ovlp)
    # mf._eri = eri
    # mf.kernel()
    # return mf


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

    # return getattr(eris, key)

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

    ENABLE_MPI_PROFILE = False
    ENABLE_SERIAL_PROFILE = False

    cc_mpi = UICCSD_MPI(mf).set(verbose=7)

    if ENABLE_MPI_PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    if rank == 0:
        print("Testing UCCSD add_vvvv ...")
    res = testfuncs.test_add_vvvv(cc_mpi)
    if ENABLE_MPI_PROFILE:
        pr.disable()
        pr.dump_stats(f'test_add_vvvv_np1.prof')
    cc_mpi = None

    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()

        if ENABLE_SERIAL_PROFILE:
            pr = cProfile.Profile()
            pr.enable()

        oovvs_ref = pyscf_uccsd._add_vvvv(cc_ref, t1_ref, t2_ref, eris=eris)

        if ENABLE_SERIAL_PROFILE:
            pr.disable()
            pr.dump_stats(f'test_add_vvvv_serial.prof')

        keys = ('oovv_aa', 'oovv_ab', 'oovv_bb')
        ref = {k: oovvs_ref[i] for i, k in enumerate(keys)}
        print("oovv results:")
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")


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


def test_update_amps(mf: UIHF, with_vector: bool = False):
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    res = testfuncs.test_vector(cc_mpi) if with_vector else testfuncs.test_update_amps(cc_mpi)
    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()
        for _ in range(3):
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
        for _ in range(3):
            l1_ref, l2_ref = pyscf_uccsd_lambda.update_lambda(
                cc_ref, t1=t1_ref, t2=t2_ref, l1=l1_ref, l2=l2_ref,
                eris=eris_ref, imds=imds_ref
        )
        (l1a, l1b), (l2aa, l2ab, l2bb) = l1_ref, l2_ref
        ref = dict(l1a=l1a, l1b=l1b, l2aa=l2aa, l2ab=l2ab, l2bb=l2bb)
        for k in res:
            diff = np.abs(res[k] - ref[k]).max()
            norm = np.linalg.norm(ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")
            assert np.allclose(diff, 0.0)


def test_ccsd_kernel(mf: UIHF):

    RUN_MPI = True
    RUN_SERIAL = False
    ENABLE_MPI_PROFILE = True
    ENABLE_SERIAL_PROFILE = False

    cc_mpi = UICCSD_MPI(mf).set(verbose=5, max_cycle=30)

    # if ENABLE_MPI_PROFILE:
    # pr = cProfile.Profile()
    # pr.enable()

    # if RUN_MPI:
    cc_mpi.kernel()

    # if ENABLE_MPI_PROFILE:
    # pr.disable()
    # pr.dump_stats(f'test_H2O_trimer_pvtz.prof')

    if RUN_SERIAL:
        cc_ref = UICCSD(mf).set(verbose=5, max_cycle=30)

    if ENABLE_SERIAL_PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    if RUN_SERIAL:
        cc_ref.kernel()

    if ENABLE_SERIAL_PROFILE:
        pr.disable()
        pr.dump_stats(f'test_H2O_trimer_serial_pvtz.prof')

    if RUN_MPI and RUN_SERIAL:
        diff = np.abs(cc_mpi.e_corr - cc_ref.e_corr)
        print(f"Difference of CCSD energy: {diff} / {np.abs(cc_ref.e_corr)} = {diff/np.abs(cc_ref.e_corr)}")
        assert np.allclose(diff, 0.0)


def test_rdm_single_shot(mf: UIHF):
    UPDATE_STEP = 3
    cc_mpi = UICCSD_MPI(mf).set(verbose=7)
    rdm_mpi = testfuncs.test_rdm_single_shot(cc_mpi)

    if rank == 0:
        cc_ref = UICCSD(mf)
        eris = cc_ref.ao2mo()
        _, t1_ref, t2_ref = cc_ref.init_amps()
        for _ in range(UPDATE_STEP):
            t1_ref, t2_ref = cc_ref.update_amps(t1_ref, t2_ref, eris=eris)
        l1_ref, l2_ref = t1_ref, t2_ref
        imds = pyscf_uccsd_lambda.make_intermediates(cc_ref, t1_ref, t2_ref, eris)
        for _ in range(UPDATE_STEP):
            l1_ref, l2_ref = pyscf_uccsd_lambda.update_lambda(
                cc_ref, t1=t1_ref, t2=t2_ref, l1=l1_ref, l2=l2_ref, eris=eris,
                imds=imds
            )
        rdm1_ref = cc_ref.make_rdm1(t1=t1_ref, t2=t2_ref,
                                    l1=l1_ref, l2=l2_ref, ao_repr=True)
        rdm2_ref = cc_ref.make_rdm2(t1=t1_ref, t2=t2_ref,
                                    l1=l1_ref, l2=l2_ref, ao_repr=True)
        rdm_ref = dict(
            t1a=t1_ref[0], t1b=t1_ref[1],
            t2aa=t2_ref[0], t2ab=t2_ref[1], t2bb=t2_ref[2],
            l1a=l1_ref[0], l1b=l1_ref[1],
            l2aa=l2_ref[0], l2ab=l2_ref[1], l2bb=l2_ref[2],
            rdm1a=rdm1_ref[0], rdm1b=rdm1_ref[1],
            rdm2aa=rdm2_ref[0], rdm2ab=rdm2_ref[1], rdm2bb=rdm2_ref[2])

        for k in rdm_mpi:
            diff = np.abs(rdm_mpi[k] - rdm_ref[k]).max()
            norm = np.linalg.norm(rdm_ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")


def test_ccsd_all(mf: UIHF, krylov: bool = False, run_rdm: bool = True):
    """
    Current stage:
    krylov amp equation correct; lambda equation not correct.
    """
    cls_mpi, cls_serial = [(UICCSD_MPI, UICCSD), (UICCSD_KRYLOV_MPI, UICCSD_KRYLOV)][krylov]

    cc_mpi = cls_mpi(mf).set(verbose=5)
    cc_mpi.kernel()
    t1_mpi, t2_mpi = cc_mpi.gather_amplitudes()

    # Should use args rather than kwargs (e.g. t1=t1_mpi, t2=t2_mpi) here.
    cc_mpi.distribute_amplitudes_(t1_mpi, t2_mpi)

    if run_rdm:
        rdm1 = cc_mpi.make_rdm1(ao_repr=True)
        rdm2 = cc_mpi.make_rdm2(ao_repr=True)
        l1_mpi, l2_mpi = cc_mpi.gather_lambda()

    res_mpi = dict(
        e_corr=cc_mpi.e_corr,
        t1=np.array(t1_mpi), t2=np.array(t2_mpi))
    if run_rdm:
        res_mpi.update(l1=np.array(l1_mpi), l2=np.array(l2_mpi),
            rdm1=np.array(rdm1), rdm2=np.array(rdm2))

    if rank == 0:
        cc_ref = cls_serial(mf).set(verbose=5)
        cc_ref.kernel()

        if run_rdm:
            rdm1_ref = cc_ref.make_rdm1(ao_repr=True)
            rdm2_ref = cc_ref.make_rdm2(t1=cc_ref.t1, t2=cc_ref.t2, l1=cc_ref.l1, l2=cc_ref.l2, ao_repr=True)

        res_ref = dict(
            e_corr=cc_ref.e_corr,
            t1=np.array(cc_ref.t1), t2=np.array(cc_ref.t2))
        
        if run_rdm:
            res_ref.update(l1=np.array(cc_ref.l1), l2=np.array(cc_ref.l2),
            rdm1=np.array(rdm1_ref), rdm2=np.array(rdm2_ref))

        for k in res_mpi:
            diff = np.abs(res_mpi[k] - res_ref[k]).max()
            norm = np.linalg.norm(res_ref[k])
            print(f"Difference of {k}: {diff} / {norm} = {diff/norm}")
            assert np.allclose(diff, 0.0, rtol=1e-8)


def test_restore(mf: UIHF):
    """
    Test whether the vector<->amplitudes conversion is correct.
    """
    cc_mpi = UICCSD_MPI(mf).set(verbose=5)
    cc_mpi.kernel()
    cc_mpi.make_rdm1(ao_repr=True)
    t1_mpi, t2_mpi = cc_mpi.gather_amplitudes()
    l1_mpi, l2_mpi = cc_mpi.gather_lambda()

    cc_mpi.save_amps(fname='./data/fcc')
    cc_mpi._release_regs()
    cc_mpi = None

    cc_mpi = UICCSD_MPI(mf).set(verbose=5)
    cc_mpi.restore_from_h5(fname='./data/fcc')
    t1_loaded, t2_loaded = cc_mpi.gather_amplitudes()
    l1_loaded, l2_loaded = cc_mpi.gather_lambda()

    t1_mpi, t2_mpi, t1_loaded, t2_loaded = map(np.array, (t1_mpi, t2_mpi, t1_loaded, t2_loaded))
    l1_mpi, l2_mpi, l1_loaded, l2_loaded = map(np.array, (l1_mpi, l2_mpi, l1_loaded, l2_loaded))

    diff_t1 = np.abs(t1_loaded - t1_mpi).max()
    diff_t2 = np.abs(t2_loaded - t2_mpi).max()
    diff_l1 = np.abs(l1_loaded - l1_mpi).max()
    diff_l2 = np.abs(l2_loaded - l2_mpi).max()
    norm_t1 = np.linalg.norm(t1_mpi)
    norm_t2 = np.linalg.norm(t2_mpi)
    norm_l1 = np.linalg.norm(l1_mpi)
    norm_l2 = np.linalg.norm(l2_mpi)

    print(f"Difference of t1: {diff_t1} / {norm_t1} = {diff_t1/norm_t1}")
    print(f"Difference of t2: {diff_t2} / {norm_t2} = {diff_t2/norm_t2}")
    print(f"Difference of l1: {diff_l1} / {norm_l1} = {diff_l1/norm_l1}")
    print(f"Difference of l2: {diff_l2} / {norm_l2} = {diff_l2/norm_l2}")
    assert np.allclose(diff_t1, 0.0)
    assert np.allclose(diff_t2, 0.0)
    assert np.allclose(diff_l1, 0.0)
    assert np.allclose(diff_l2, 0.0)


if __name__ == "__main__":
    from mpi4pyscf.cc.test.conftest import H2O_mol, H2O_trimer_mol, H2O_dimer_mol
    import sys

    task = 'uccsd'
    if len(sys.argv) > 1:
        task = sys.argv[1]
    if len(sys.argv) < 3: chkpt = None
    else: chkpt = int(sys.argv[2])

    # mf = get_scf_solver_mol(mol=H2O_trimer_mol(), chkfile='./data/chk_h2o_trimer_uhf.h5')
    # mf = get_scf_solver_O2()
    mf = get_scf_solver_mol(mol=H2O_mol())

    if task == "eris": test_uiccsd_eris(mf)
    elif task == "init_amps": test_init_amps(mf)
    elif task == "tau": test_make_tau(mf)
    elif task == "vvvv": test_add_vvvv(mf)
    elif task == 'energy': test_energy(mf)
    elif task == "kernel": test_ccsd_kernel(mf)
    elif task == "rdm": test_rdm_single_shot(mf)
    elif task == 'uccsd': test_ccsd_all(mf, krylov=False, run_rdm=True)
    elif task == 'krylov': test_ccsd_all(mf, krylov=True, run_rdm=True)
    elif task == "restore": test_restore(mf)

    elif task == "amps":
        if chkpt is None: test_update_amps(mf)
        else: test_update_amps_checkpoint(mf, checkpoint=chkpt)

    elif task == "vector":
        test_update_amps(mf, with_vector=True)

    elif task == "imds":
        if chkpt is None: test_lambda_intermediates(mf)
        else: test_lambda_intermediates_checkpoint(mf, checkpoint=chkpt)

    elif task == "lambda":
        if chkpt is None: test_update_lambda(mf=mf)
        else: test_update_lambda_checkpoint(mf, checkpoint=chkpt)

    else:
        raise ValueError(f"Unknown task {task}.")
