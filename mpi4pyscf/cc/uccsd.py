"""
MPI-UCCSD with real integrals.
"""
from functools import reduce
import numpy as np
from numpy.typing import ArrayLike

from pyscf import lib, ao2mo, __config__
from pyscf.cc import uccsd as pyscf_uccsd

from libdmet.utils.misc import take_eri, tril_take_idx

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.ccsd import _sync_, _pack_scf, _task_location
from mpi4pyscf.cc import cc_tools as tools
from mpi4pyscf.cc.cc_tools import SegArray


comm = mpi.comm
rank = mpi.rank
ntasks = mpi.pool.size


@mpi.parallel_call(skip_args=[1], skip_kwargs=['eris'])
def init_amps(mycc, eris=None):
    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo()
        eris = mycc._eris

    time0 = logger.process_clock(), logger.perf_counter()
    nocca, noccb = mycc.nocc
    nmoa, nmob = map(np.size, eris.mo_energy)
    nvira, nvirb = nmoa - nocca, nmob - noccb

    # print(eris.focka.shape)
    # print(f"nocca = {nocca}, noccb = {noccb}, nmoa = {nmoa}, nmob = {nmob}")
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:]
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:]
    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    t1a = fova.conj() / eia_a
    t1b = fovb.conj() / eia_b

    (vloc0a, vloc1a), (vloc0b, vloc1b) = map(_task_location, (nvira, nvirb))

    vlocsa, vlocsb = map(tools.get_vlocs, (nvira, nvirb))

    eris_oxov, eris_OXOV, eris_oxOV = [getattr(eris, k) for k in ('oxov', 'OXOV', 'oxOV')]

    # print(f"shapes: {eris_oxov.shape}, {eris_OXOV.shape}, {eris_oxOV.shape}")

    t2_ooxv = eris_oxov.transpose(0, 2, 1, 3) / lib.direct_sum("ix+jb->ijxb", eia_a[:, vloc0a:vloc1a], eia_a)
    t2_OOXV = eris_OXOV.transpose(0, 2, 1, 3) / lib.direct_sum("iX+JB->iJXB", eia_b[:, vloc0b:vloc1b], eia_b)
    t2_oOxV = eris_oxOV.transpose(0, 2, 1, 3) / lib.direct_sum("ix+JB->iJxB", eia_a[:, vloc0a:vloc1a], eia_b)

    t2_ooxv = tools.segmented_transpose(t2_ooxv, 2, 3, 1., -1., vlocs=vlocsa)
    t2_OOXV = tools.segmented_transpose(t2_OOXV, 2, 3, 1., -1., vlocs=vlocsb)

    emp2 = np.einsum("iJxB,ixJB->", t2_oOxV, eris_oxOV)
    emp2 += np.einsum("ijxb,ixjb->", t2_ooxv, eris_oxov) * .5
    emp2 += np.einsum("IJXB,IXJB->", t2_OOXV, eris_OXOV) * .5
    mycc.emp2 = comm.allreduce(emp2)

    logger.info(mycc, "Init t2, MP2 energy = %.15g", mycc.emp2)
    logger.timer(mycc, "UICCSD init amplitudes", *time0)
    return mycc.emp2, (t1a, t1b), (t2_ooxv, t2_oOxV, t2_OOXV)


def make_tau_aa(t2_ooxv, t1a, r1a, vlocs, fac=1., out=None):
    """
    t1a, r1a are broadcasted arrays while t2_ooxv is a segmented array.
    return the segmented tau_aa at index 2.
    """
    vloc0, vloc1 = vlocs[rank]
    tau1aa = np.einsum("ix,jb->ijxb", t1a[:, vloc0:vloc1], r1a)
    tau1aa -= np.einsum("ix,jb->jixb", t1a[:, vloc0:vloc1], r1a)
    tau1aa = tools.segmented_transpose(tau1aa, 2, 3, 1., -1., vlocs=vlocs)
    tau1aa *= fac * .5
    tau1aa += t2_ooxv
    return tau1aa


def make_tau_ab(t2_oOxV, t1, r1, vlocs, fac=1., out=None):
    (t1a, t1b), (r1a, r1b) = t1, r1
    vloc0a, vloc1a = vlocs[rank]
    tau1ab = np.einsum('ix,jb->ijxb', t1a[:, vloc0a:vloc1a], r1b)
    tau1ab += np.einsum('ix,jb->ijxb', r1a[:, vloc0a:vloc1a], t1b)
    tau1ab *= (fac * .5)
    tau1ab += t2_oOxV
    return tau1ab


def make_tau(t2, t1, r1, vlocs, fac=1., out=None) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    (t1a, t1b), (r1a, r1b) = t1, r1
    t2_ooxv, t2_oOxV, t2_OOXV = t2
    vlocsa, vlocsb = vlocs
    tau_ooxv = make_tau_aa(t2_ooxv, t1a, r1a, vlocsa, fac=fac, out=out)
    tau_OOXV = make_tau_aa(t2_OOXV, t1b, r1b, vlocsb, fac=fac, out=out)
    tau_oOxV = make_tau_ab(t2_oOxV, t1, r1, vlocsa, fac=fac, out=out)
    return tau_ooxv, tau_oOxV, tau_OOXV


def _contract_vvvv_t2(h_vxvv, t_ooxv):
    # currently slow version
    _einsum = tools.segmented_einsum_new
    t_oxv = lib.pack_tril(t_ooxv.transpose(2, 3, 0, 1)).transpose(2, 0, 1)
    res = _einsum("acbd,icd->iab", arrs=(h_vxvv, t_oxv), seg_idxs=(1, 1))
    return lib.unpack_tril(res.transpose(1, 2, 0)).transpose(2, 3, 0, 1)


def _contract_vvVV_t2(h_vxVV, t_oOxV):
    _einsum = tools.segmented_einsum_new
    return _einsum("ayBD,iJyD->iJaB", arrs=(h_vxVV, t_oOxV), seg_idxs=(1, 2))


def _add_vvvv(t1, t2, eris, vlocs: tuple):
    if t1 is None:
        t2_ooxv, t2_oOxV, t2_OOXV = t2
    else:
        t2_ooxv, t2_oOxV, t2_OOXV = make_tau(t2, t1, t1, vlocs=vlocs)

    u2_ooxv = _contract_vvvv_t2(eris.vxvv, t2_ooxv)
    u2_OOXV = _contract_vvvv_t2(eris.VXVV, t2_OOXV)
    u2_oOxV = _contract_vvVV_t2(eris.vxVV, t2_oOxV)

    return u2_ooxv, u2_oOxV, u2_OOXV


class _ChemistsERIs:
    _eri_keys = ('oooo', 'oxoo', 'oxov', 'ooxv', 'ovxo', 'oxvv', 'vxvv',
                 'OOOO', 'OXOO', 'OXOV', 'OOXV', 'OVXO', 'OXVV', 'VXVV',
                 'ooOO', 'oxOO', 'oxOV', 'ooXV', 'oxVO', 'oxVV', 'vxVV',
                 'OXoo', 'OOxv', 'OVxo', 'OXvv')

    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        for k in self._eri_keys:
            setattr(self, k, None)


    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = mycc.get_frozen_mask()
        self.mo_coeff = mo_coeff = \
                (mo_coeff[0][:,mo_idx[0]], mo_coeff[1][:,mo_idx[1]])
        # Note: Recomputed fock matrix since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.focka = reduce(np.dot, (mo_coeff[0].conj().T, fockao[0], mo_coeff[0]))
        self.fockb = reduce(np.dot, (mo_coeff[1].conj().T, fockao[1], mo_coeff[1]))
        self.fock = (self.focka, self.fockb)

        nocca, noccb = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_ea = self.focka.diagonal().real
        mo_eb = self.fockb.diagonal().real
        self.mo_energy = (mo_ea, mo_eb)
        gap_a = abs(mo_ea[:nocca,None] - mo_ea[None,nocca:])
        gap_b = abs(mo_eb[:noccb,None] - mo_eb[None,noccb:])
        if gap_a.size > 0:
            gap_a = gap_a.min()
        else:
            gap_a = 1e9
        if gap_b.size > 0:
            gap_b = gap_b.min()
        else:
            gap_b = 1e9
        if gap_a < 1e-5 or gap_b < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap (%s,%s) too small for UCCSD',
                        gap_a, gap_b)
        return self

    def get_integral(self, key):
        """
        Get the unravelled eris.
        """
        res = getattr(self, key)
        if key not in ('oxvv', 'vxvv', 'OXVV', 'VXVV', 'OXvv', 'oxVV', 'vxVV'):
            return res
        # for the segment with ?xvv, unpack the triled indices of the last two indices.
        assert res.ndim == 3
        n1, n2, nvir_pair = res.shape
        r = lib.unpack_tril(res.reshape(n1 * n2, nvir_pair))
        nvir = r.shape[2]
        return r.reshape(n1, n2, nvir, nvir)


def _preprocess_eri_uihf(eri_ao):
    if isinstance(eri_ao, np.ndarray):
        if eri_ao.ndim == 1: # 8-fold
            eri_ao = [eri_ao, eri_ao, eri_ao]
        elif eri_ao.ndim == 2:
            if len(eri_ao) == 1:
                eri_ao = [eri_ao[0], eri_ao[0], eri_ao[0]]
            else: # 4-fold
                eri_ao = [eri_ao, eri_ao, eri_ao]
        elif eri_ao.ndim == 3:
            if len(eri_ao) == 1:
                eri_ao = [eri_ao[0], eri_ao[0], eri_ao[0]]
        elif eri_ao.ndim == 4:
            eri_ao = [eri_ao, eri_ao, eri_ao]
        elif eri_ao.ndim == 5:
            if len(eri_ao) == 1:
                eri_ao = [eri_ao[0], eri_ao[0], eri_ao[0]]
        else:
            raise ValueError("Unknown ERI shape %s"%(str(eri_ao.shape)))
    else:
        if len(eri_ao) == 1:
            eri_ao = [eri_ao[0], eri_ao[0], eri_ao[0]]
        elif len(eri_ao) == 3:
            eri_ao = eri_ao
        else:
            raise ValueError("Unknown ERI length %s"%(len(eri_ao)))
    return eri_ao


def _make_eriaa_incore(eri: ArrayLike, eris: _ChemistsERIs, nocc: int,
                       nmo: int, spin=0, log=None, cput0=None):
    nvir = nmo - nocc
    vlocs = tools.get_vlocs(nvir)

    occ_range_dict = dict(o=np.arange(0, nocc), v=np.arange(nocc, nmo))
    eri_tags = ('oooo', 'ovoo', 'ovov', 'oovv', 'ovvo', 'ovvv', 'vvvv')
    eri_seg_tags = [('oooo', 'oxoo', 'oxov', 'ooxv', 'ovxo', 'oxvv', 'vxvv'),
                    ('OOOO', 'OXOO', 'OXOV', 'OOXV', 'OVXO', 'OXVV', 'VXVV')][spin]
    slice_idx = [None, 1, 1, 2, 2, 1, 1]

    cput = cput0
    for eri_tag, eri_seg_tag, seg_idx in zip(eri_tags, eri_seg_tags, slice_idx):
        def _fn():
            idx_ov = tril_take_idx(occ_range_dict['o'], occ_range_dict['v'], compact=False)
            idx_vv_nt = tril_take_idx(occ_range_dict['v'], occ_range_dict['v'], compact=False)
            idx_vv = tril_take_idx(occ_range_dict['v'], occ_range_dict['v'], compact=True)
            if eri_tag == "ovvv":
                res = eri[np.ix_(idx_ov, idx_vv)].reshape(nocc, nvir, nvir*(nvir+1)//2)
            elif eri_tag == "vvvv":
                res = eri[np.ix_(idx_vv_nt, idx_vv)].reshape(nvir, nvir, nvir*(nvir+1)//2)
            else:
                res = take_eri(eri, *map(occ_range_dict.get, eri_tag), compact=(eri_tag == 'vvvv'))
            # print(f"In _make_eriaa_incore: {eri_tag} shape = {res.shape}")
            return res

        if rank == 0 and log.verbose >= logger.DEBUG:
            print(f"Seg_idx = {seg_idx} for eri_tag = {eri_tag}")
        eri_mo_seg = tools.get_mpi_array(_fn, vlocs=vlocs, seg_idx=seg_idx)
        setattr(eris, eri_seg_tag, eri_mo_seg)
        cput = log.timer("UICCSD scatter {}:              ".format(eri_seg_tag), *cput)
    return cput


def _make_eriab_incore(eri_ab: ArrayLike, eris: _ChemistsERIs,
                       nocc: tuple, nmo: tuple, log=None, cput0=None):
    nmoa, nmob = nmo
    nocca, noccb = nocc
    nvira, nvirb = nmoa - nocca, nmob - noccb
    vlocs_a, vlocs_b = map(tools.get_vlocs, (nvira, nvirb))

    occ_range_dict = dict(o=np.arange(0, nocca), v=np.arange(nocca, nmoa),
                          O=np.arange(0, noccb), V=np.arange(noccb, nmob))

    eri_tags = ('ooOO', 'ovOO', 'ovOV', 'ooVV', 'ovVO', 'ovVV', 'vvVV',
                'OVoo', 'OOvv', 'OVvo', 'OVvv')
    eri_seg_tags = ('ooOO',
                    'oxOO', 'oxOV', 'ooXV', 'oxVO', 'oxVV',
                    'vxVV', 'OXoo', 'OOxv', 'OVxo', 'OXvv')
    slice_idx = [None, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1]

    cput = cput0

    for eri_tag, eri_seg_tag, seg_idx in zip(eri_tags, eri_seg_tags, slice_idx):
        def _fn():
            idx_ov = tril_take_idx(occ_range_dict['o'], occ_range_dict['v'], compact=False)
            idx_OV = tril_take_idx(occ_range_dict['O'], occ_range_dict['V'], compact=False)
            idx_vv_nt = tril_take_idx(occ_range_dict['v'], occ_range_dict['v'], compact=False)
            idx_vv = tril_take_idx(occ_range_dict['v'], occ_range_dict['v'], compact=True)
            idx_VV = tril_take_idx(occ_range_dict['V'], occ_range_dict['V'], compact=True)
            eri_ba = eri_ab.T
            if eri_tag == 'ovVV':
                return eri_ab[np.ix_(idx_ov, idx_VV)].reshape(nocca, nvira, nvirb*(nvirb+1)//2)
            elif eri_tag == "OVvv":
                return eri_ba[np.ix_(idx_OV, idx_vv)].reshape(noccb, nvirb, nvira*(nvira+1)//2)
            elif eri_tag == 'vvVV':
                return eri_ab[np.ix_(idx_vv_nt, idx_VV)].reshape(nvira, nvira, nvirb*(nvirb+1)//2)
            elif eri_tag in ('OVoo', 'OOvv', 'OVvo'):
                return take_eri(eri_ba, *map(occ_range_dict.get, eri_tag), compact=False)
            else:
                return take_eri(eri_ab, *map(occ_range_dict.get, eri_tag), compact=(eri_tag == 'vvVV'))

        if rank == 0 and log.verbose >= logger.DEBUG:
            print(f"Seg_idx = {seg_idx} for eri_tag = {eri_tag}")

        eri_mo_seg = tools.get_mpi_array(_fn, vlocs=[vlocs_a, vlocs_b]['X' in eri_seg_tag], seg_idx=seg_idx)
        setattr(eris, eri_seg_tag, eri_mo_seg)
        cput = log.timer("UICCSD scatter {}:              ".format(eri_seg_tag), *cput)
    return cput


@mpi.parallel_call
def _make_eris_incore_uihf(mycc, mo_coeff=None, ao2mofn=None):
    """
    For the ERI with aa/ab/bb part, get the blocked ERI in MO basis for CCSD solver.
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    _sync_(mycc)
    eris = _ChemistsERIs(mol=None)

    if rank == 0:
        assert ao2mofn is None
        eris._common_init_(mycc, mo_coeff)
        comm.bcast((eris.mo_coeff, eris.focka, eris.fockb, eris.nocc, eris.mo_energy))
    else:
        eris.mol = mycc.mol
        eris.mo_coeff, eris.focka, eris.fockb, eris.nocc, eris.mo_energy = comm.bcast(None)

    (nocca, noccb), (nmoa, nmob) = mycc.nocc, mycc.nmo
    moa, mob = eris.mo_coeff
    (naoa, nmoa), (naob, nmob) = moa.shape, mob.shape
    assert naoa == naob

    eri_ao, eri_aa, eri_bb, eri_ab = None, None, None, None

    if rank == 0:
        eri_ao = mycc._scf._eri
        eri_ao = _preprocess_eri_uihf(eri_ao)
        eri_aa = ao2mo.full(ao2mo.restore(4, eri_ao[0], naoa), moa, compact=True)
    comm.Barrier()
    cput = log.timer("UICCSD ao2mo(aa):             ", *cput0)
    cput = _make_eriaa_incore(eri_aa, eris, nocca, nmoa, spin=0, log=log, cput0=cput)
    eri_aa = None

    if rank == 0:
        eri_bb = ao2mo.full(ao2mo.restore(4, eri_ao[1], naob), mob, compact=True)
    comm.Barrier()
    cput = log.timer("UICCSD ao2mo(bb):             ", *cput)
    cput = _make_eriaa_incore(eri_bb, eris, noccb, nmob, spin=1, log=log, cput0=cput)
    eri_bb = None

    if rank == 0:
        eri_ab = ao2mo.general(ao2mo.restore(4, eri_ao[2], naoa),
                               (moa, moa, mob, mob), compact=True)
    comm.Barrier()
    cput = log.timer("UICCSD ao2mo(ab):             ", *cput)
    cput = _make_eriab_incore(eri_ab, eris, (nocca, noccb), (nmoa, nmob), log=log, cput0=cput)
    eri_ab = None

    mycc._eris = eris
    log.timer('UICCSD integral transformation   ', *cput0)
    return eris


def update_amps(cc, t1, t2, eris):

    log = logger.Logger(cc.stdout, cc.verbose)

    t1a, t1b = t1
    t2_ooxv, t2_oOxV, t2_OOXV = t2  # aa, ab, bb
    (nocca, nvira), (noccb, nvirb) = t1a.shape, t1b.shape
    nvira_seg, nvirb_seg = t2_ooxv.shape[2], t2_OOXV.shape[2]
    vlocs_a, vlocs_b = map(tools.get_vlocs, (nvira, nvirb))
    vlocs_ab = (vlocs_a, vlocs_b)
    (vloc0a, vloc1a), (vloc0b, vloc1b) = vlocs_a[rank], vlocs_b[rank]
    slc_va, slc_vb = slice(vloc0a, vloc1a), slice(vloc0b, vloc1b)
    dtype = t1a.dtype


    def _fntp(arr: SegArray, idx1: int, idx2: int, coeff_0: float, coeff: float) -> SegArray:
        """wrapped transpose function for SegArray"""
        if idx1 != arr.seg_idx and idx2 != arr.seg_idx: # neither idx1 nor idx2 is the seg_idx
            idxmin, idxmax = min(idx1, idx2), max(idx1, idx2)
            tp = tuple(range(idxmin)) + (idxmax, ) \
                    + tuple(range(idxmin+1, idxmax)) + (idxmin,) \
                    + tuple(range(idxmax+1, arr.data.ndim))
            return SegArray(arr.data.transpose(tp) * coeff + arr.data * coeff_0, arr.seg_idx, arr.seg_spin)
        else:
            vlocs = vlocs_a if arr.seg_spin == 0 else vlocs_b
            if idx1 == arr.seg_idx:
                arr = tools.segmented_transpose(arr.data, idx1, idx2, coeff_0, coeff, vlocs=vlocs)
            elif idx2 == arr.seg_idx:
                arr = tools.segmented_transpose(arr.data, idx2, idx1, coeff_0, coeff, vlocs=vlocs)
            return SegArray(arr, arr.seg_idx, arr.seg_spin)

    def _einsum(subscripts: str, arr1: SegArray, arr2: SegArray, out: SegArray | None = None) -> SegArray:
        """
        wrapped einsum function
        out: the output SegArray to be added on,
            may be used when segmented einsum type == 'outer',
            so that the re-segmentation can be avoided.
        """
        vlocs_1 = [vlocs_a, vlocs_b][arr1.seg_spin] if arr1.seg_spin is not None else None
        vlocs_2 = [vlocs_a, vlocs_b][arr2.seg_spin] if arr2.seg_spin is not None else None
        seg_idxs = (arr1.seg_idx, arr2.seg_idx)
        final_spin = None

        if out is not None:
            seg_idxs += (out.seg_idx, )
            final_spin = out.seg_spin

        final_data, final_seg_idx = tools.segmented_einsum_new(
            subscripts=subscripts,
            arrs=(arr1.data, arr2.data), seg_idxs=seg_idxs,
            vlocss=(vlocs_1, vlocs_2), return_output_idx=True
            )
        return SegArray(final_data, seg_idx=final_seg_idx, seg_spin=final_spin)

    def _get_integral(key: str) -> SegArray:
        seg_idx, seg_spin = None, None
        arr_label = "eris." + key.replace('x', 'v').replace('X', 'V')
        for k in 'xX':
            if k in key:
                seg_idx, seg_spin = key.index(k), 'xX'.index(k)
                break
        return SegArray(eris.get_integral(key), seg_idx, seg_spin, label=arr_label)

    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + cc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + cc.level_shift
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]

    Fooa = np.zeros((nocca, nocca), dtype=dtype)
    Foob = np.zeros((noccb, noccb), dtype=dtype)
    Fvva = np.zeros((nvira, nvira), dtype=dtype)
    Fvvb = np.zeros((nvirb, nvirb), dtype=dtype)
    u1a = np.zeros_like(t1a)
    u1b = np.zeros_like(t1b)

    taus = make_tau(t2, t1, t1, (vlocs_a, vlocs_b))
    tau_ooxv, tau_oOxV, tau_OOXV = taus
    u2_ooxv, u2_oOxV, u2_OOXV = _add_vvvv(t1=None, t2=taus, eris=eris, vlocs=vlocs_ab)
    u2_ooxv *= .5
    u2_OOXV *= .5


    # Build the segmented array of the intermediates.

    ## Transform the broadcaseted 1e arrays to SegArray
    h1_labels = ('t1a', 't1b', 'u1a', 'u1b', 'Fooa', 'Foob', 'Fvva', 'Fvvb')
    t1a, t1b, u1a, u1b, Fooa, Foob, Fvva, Fvvb = [SegArray(data=d, label=l)
        for d, l in zip((t1a, t1b, u1a, u1b, Fooa, Foob, Fvva, Fvvb), h1_labels)]

    ## Transform the 2e arrays to SegArray
    seg_spins = (0, 0, 1, 0, 0, 1, 0, 0, 1)
    t2_labels = ('t2aa', 't2ab', 't2ab', 'u2aa', 'u2ab', 'u2ab', 'tau_aa', 'tau_ab', 'tau_bb')
    t2_ooxv, t2_oOxV, t2_OOXV, u2_ooxv, u2_oOxV, u2_OOXV, tau_ooxv, tau_oOxV, tau_OOXV = [
        SegArray(data=d, seg_idx=2, seg_spin=ss, label=l) for d, ss, l in zip(
            (t2_ooxv, t2_oOxV, t2_OOXV, u2_ooxv, u2_oOxV, u2_OOXV, tau_ooxv, tau_oOxV, tau_OOXV), seg_spins, t2_labels)]

    wovxo = SegArray(np.zeros((nocca, nvira, nvira_seg, nocca), dtype=dtype), seg_idx=2, seg_spin=0, label='wovvo')
    wOVXO = SegArray(np.zeros((noccb, nvirb, nvirb_seg, noccb), dtype=dtype), seg_idx=2, seg_spin=1, label='wOVVO')
    woVxO = SegArray(np.zeros((nocca, nvirb, nvira_seg, noccb), dtype=dtype), seg_idx=2, seg_spin=0, label='woVvO')
    woVXo = SegArray(np.zeros((nocca, nvirb, nvirb_seg, nocca), dtype=dtype), seg_idx=2, seg_spin=1, label='woVVo')
    wOvXo = SegArray(np.zeros((noccb, nvira, nvirb_seg, nocca), dtype=dtype), seg_idx=2, seg_spin=1, label='wOvVo')
    wOvxO = SegArray(np.zeros((noccb, nvira, nvira_seg, noccb), dtype=dtype), seg_idx=2, seg_spin=0, label='wOvvO')


    # Contractions
    if nvira > 0 and nocca > 0:
        oxvv = _get_integral('oxvv')
        oxvv = _fntp(oxvv, 1, 3, 1.0, -1.0)
        Fvva += _einsum('mf,mfae->ae', t1a[:, slc_va], oxvv)
        wovxo += _einsum('jf,mebf->mbej', t1a, oxvv)
        u1a += .5 * _einsum('mief,meaf->ia', t2_ooxv, oxvv)
        u2_ooxv += _einsum('ie,mbea->imab', t1a, oxvv.conj())
        tmp1aa = SegArray(np.zeros((nocca, nocca, nocca, nvira), dtype=dtype), label='tmp1aa')
        tmp1aa += _einsum('ijef,mebf->ijmb', tau_ooxv, oxvv)
        u2_ooxv -= _einsum('ijmb,ma->ijab', tmp1aa, t1a[:, slc_va]*.5)
        oxvv = tmp1aa = None

    if nvirb > 0 and noccb > 0:
        OXVV = _get_integral('OXVV')
        OXVV = _fntp(OXVV, 1, 3, 1.0, -1.0)
        Fvvb += _einsum('mf,mfae->ae', t1b[: slc_vb], OXVV)
        wOVXO += _einsum('jf,mebf->mbej', t1b, OXVV)
        u1b += 0.5 * _einsum('MIEF,MEAF->IA', t2_OOXV, OXVV)
        u2_OOXV += _einsum('ie,mbea->imab', t1b, OXVV.conj())
        tmp1bb = SegArray(np.zeros((noccb, noccb, noccb, nvirb), dtype=dtype), label='tmp1bb')
        tmp1bb += _einsum('ijef,mebf->ijmb', tau_OOXV, OXVV)
        u2_OOXV -= _einsum('ijmb,ma->ijab', tmp1bb, t1b[:, slc_vb] * .5)
        OXVV = tmp1bb = None

    if nvirb > 0 and nocca > 0:
        oxVV = _get_integral('oxVV')
        Fvvb += _einsum('mf,mfAE->AE', t1a[:, slc_va], oxVV)
        woVxO += _einsum('JF,meBF->mBeJ', t1b, oxVV)
        woVXo += _einsum('jf,mfBE->mBEj',-t1a[:, slc_va], oxVV)
        u1b += _einsum('mIeF,meAF->IA', t2_oOxV, oxVV)
        u2_oOxV += lib.einsum('IE,maEB->mIaB', t1b, oxVV.conj())
        tmp1ab = SegArray(np.zeros((nocca, noccb, nocca, nvirb), dtype=dtype), label='tmp1ab')
        tmp1ab += _einsum('iJeF,meBF->iJmB', tau_oOxV, oxVV)
        u2_oOxV -= _einsum('iJmB,ma->iJaB', tmp1ab, t1a[:, slc_va])
        oxVV = tmp1ab = None

    if nvira > 0 and noccb > 0:
        OXvv = _get_integral('OXvv')
        Fvva += _einsum('MF,MFae->ae', t1b[:, slc_vb], OXvv)
        wOvXo += _einsum('jf,MEaf->MaEj', t1a, OXvv)
        wOvxO += _einsum('JF,MFbe->MbeJ', -t1b[:, slc_vb], OXvv)
        u1a += _einsum('iMfE,MEaf->ia', t2_oOxV, OXvv)
        u2_oOxV += _einsum('ie,MBea->iMaB', t1a, OXvv.conj())
        tmp1ba = SegArray(np.zeros((noccb, nocca, nvirb, nocca), dtype=dtype), label='tmp1ba')
        tmp1ba += _einsum('iJeF,MFbe->iJbM', tau_oOxV, OXvv)
        u2_oOxV -= _einsum('iJbM,MA->iJbA', tmp1ba, t1b[:, slc_vb])
        OXvv = tmp1ba = None

    oxov, oxoo = map(_get_integral, ('oxov', 'oxoo'))
    Woooo = SegArray(np.zeros((nocca, ) * 4, dtype=dtype), label='Woooo')
    Woooo = _fntp(Woooo, 2, 3, 1.0, -1.0)
    Woooo += _einsum('ijef,menf->mnij', tau_ooxv, oxov) * .5
    Woooo = Woooo.collect()
    Woooo += eris.oooo.transpose(0, 2, 1, 3)
    u2_ooxv += _einsum('mnab,mnij->ijab', tau_ooxv, Woooo*.5)
    Woooo = tau_ooxv = None
    oxoo = oxoo - oxoo.transpose(2, 1, 0, 3)
    Fooa += _einsum("ne,nemi->mi", t1a[:, slc_va], oxoo)
    # u1_ox: part of u1a where virtual index 'a' is segmented.
    u1_ox = SegArray(np.zeros((nocca, nvira_seg)), seg_idx=1, seg_spin=0, label='u1_ox')
    u1_ox += .5 * _einsum('mnae,meni->ia', t2_ooxv, oxoo)
    wovxo += _einsum('nb,nemj->mbej', t1a, oxoo)
    oxoo = None

    til_ooxv = make_tau_aa(t2_ooxv.data, t1a.data, t1a.data, vlocs=vlocs_a, fac=0.5)
    til_ooxv = SegArray(til_ooxv, seg_idx=2, seg_spin=0, label='til_aa')
    oxov = _fntp(oxov, 1, 3, 1.0, -1.0)
    F_xv = SegArray(np.zeros((nvira_seg, nvira), dtype=dtype), seg_idx=0, seg_spin=0, label='F_xv')
    F_xv -= .5 * _einsum("mnaf,menf->ae", til_ooxv, oxov)
    Fooa -= .5 * _einsum('inef,menf->mi', til_ooxv, oxov)
    F_ox = SegArray(np.zeros((nocca, nvira_seg), dtype=dtype), seg_idx=1, seg_spin=0, label='F_ox')
    F_ox += _einsum('nf,menf->me',t1a, oxov)
    u2_ooxv += oxov.conj().transpose(0, 2, 1, 3) * .5
    wovxo -= .5 * _einsum('jnfb,menf->mbej', t2_ooxv, oxov)
    woVxO += .5 * _einsum('nJfB,menf->mBeJ', t2_oOxV, oxov)

    tmpaa = _einsum('jf,menf->mnej', t1a, oxov).set(label='tmpaa')
    wovxo -= _einsum('nb,mnej->mbej', t1a, tmpaa)
    oxov = tmpaa = til_ooxv = None

    OXOV, OXOO = map(_get_integral, ('OXOV', 'OXOO'))
    WOOOO = _einsum('je,nemi->mnij', t1b[:, slc_vb], OXOO).set(label='WOOOO')
    WOOOO = _fntp(WOOOO, 2, 3, 1.0, -1.0)
    WOOOO += _einsum('ijef,menf->mnij', tau_OOXV, OXOV) * .5
    WOOOO = WOOOO.collect()
    WOOOO += eris.OOOO.transpose(0, 2, 1, 3)
    u2_OOXV += _einsum('mnab,mnij->ijab', tau_OOXV, WOOOO*.5)
    WOOOO = tau_OOXV = None
    OXOO = OXOO - OXOO.transpose(2, 1, 0, 3)
    Foob += _einsum('ne,nemi->mi', t1b[:, slc_vb], OXOO)
    u1_OX = .5 * _einsum('mnae,meni->ia', t2_OOXV, OXOO).set(label='u1_OX')
    wOVXO += _einsum('nb,nemj->mbej', t1b, OXOO)
    OXOO = None

    til_OOXV = make_tau_aa(t2_OOXV.data, t1b.data, t1b.data, vlocs=vlocs_b, fac=0.5)
    til_OOXV = SegArray(til_OOXV, seg_idx=2, seg_spin=1, label='til_bb')
    OXOV = _fntp(OXOV, 1, 3, 1.0, -1.0)
    F_XV = -.5 * _einsum('MNAF,MENF->AE', til_OOXV, OXOV).set(label='F_XV')
    Foob += .5 * _einsum('inef,menf->mi', til_OOXV, OXOV)
    F_OX = _einsum('nf,menf->me', t1b, OXOV).set(label='F_OX')
    u2_OOXV += OXOV.conj().transpose(0,2,1,3) * .5
    wOVXO -= 0.5 * _einsum('jnfb,menf->mbej', t2_OOXV, OXOV)
    wOvXo += 0.5 * _einsum('jNbF,MENF->MbEj', t2_oOxV, OXOV)
    tmpbb = _einsum('jf,menf->mnej', t1b, OXOV).set(label='tmpbb')
    wOVXO -= _einsum('nb,mnej->mbej', t1b, tmpbb)
    OXOV = tmpbb = til_OOXV = None

    OXoo, oxOO = map(_get_integral, ('OXoo', 'oxOO'))
    Fooa += _einsum('NE,NEmi->mi', t1b[:, slc_vb], OXoo)
    u1_ox -= _einsum('nMaE,MEni->ia', t2_oOxV, OXoo)
    wOvXo -= _einsum('nb,MEnj->MbEj', t1a, OXoo)
    woVXo += _einsum('NB,NEmj->mBEj', t1b, OXoo)
    Foob += _einsum('ne,neMI->MI', t1a[:, slc_va], oxOO)
    u1b -= _einsum('mNeA,meNI->IA', t2_oOxV, oxOO)
    woVxO -= _einsum('NB,meNJ->mBeJ', t1b, oxOO)
    wOvxO += _einsum('nb,neMJ->MbeJ', t1a, oxOO)
    WoOoO = _einsum('JE,NEmi->mNiJ', t1b[:, slc_vb], OXoo).set(label='WoOoO')
    WoOoO += lib.einsum('je,neMI->nMjI', t1a[:, slc_va], oxOO)
    OXoo = oxOO = None

    WoOoO += _einsum('iJeF,meNF->mNiJ', tau_oOxV, oxOV)
    WoOoO = WoOoO.collect()
    WoOoO += eris.ooOO.transpose(0, 2, 1, 3)

    oxOV = _get_integral('oxOV')
    u2_oOxV += _einsum('mNaB,mNiJ->iJaB', tau_oOxV, WoOoO)
    WoOoO = None

    til_oOxV = make_tau_ab(t2_oOxV.data, t1.data, t1.data, vlocs=vlocs_a, fac=0.5)
    F_xv -= _einsum('mNaF,meNF->ae', til_oOxV, oxOV)
    Fvvb -= _einsum('nMfA,nfME->AE', til_oOxV, oxOV)
    Fooa += _einsum('iNeF,meNF->mi', til_oOxV, oxOV)
    Foob += _einsum('nIfE,nfME->MI', til_oOxV, oxOV)
    F_ox += np.einsum('NF,meNF->me', t1b, oxOV)
    Fovb -= _einsum('nf,nfME->ME', t1a[:, slc_va], oxOV)
    til_oOxV = None

    u2_oOxV = u2_oOxV.collect()
    u2_oOxV += oxOV.conj().transpose(0, 2, 1, 3)
    wovxo += .5 * _einsum('jNbF,meNF->mbej', t2_oOxV, oxOV, out=wovxo)  # outer
    wOvxO += .5 * _einsum('nJbF,neMF->MbeJ', t2_oOxV, oxOV, out=wOvxO)

    # Whether this way is correct? Should add the auxiliary array [x],[x]y->y to einsum type.
    # FIXME NOT correct, since the two slices uses the same rank, but instead they should be independent.
    oxOY = SegArray(oxOV.data[:, :, :, slc_vb], seg_idx=1, seg_spin=0, label='oxOY')
    wOVXO -= .5 * _einsum('nJfB,nfME->MBEJ', t2_oOxV, oxOY).collect() # [x],[x]y->
    wOvXo -= .5 * _einsum('jnfb,nfME->MbEj', t2_oOxV, oxOY).collect()
    woVXo += .5 * _einsum('jNfB,mfNE->mBEj', t2_oOxV, oxOY).collect()
    woVxO -= .5 * _einsum('JNFB,meNF->mBeJ', t2_OOXV, oxOV)

    tmpabab = _einsum('JF,meNF->mNeJ', t1b, oxOV).set(label='tmpabab')
    tmpbaba = _einsum('jf,nfME->MnEj', t1a[:, slc_va], oxOY).collect().set(label='tmpbaba', seg_idx=2, seg_spin=1)

    woVxO -= _einsum('NB,mNeJ->mBeJ', t1b, tmpabab)
    wOvXo -= _einsum('nb,MnEj->MbEj', t1a, tmpbaba)
    woVXo += _einsum('NB,NmEj->mBEj', t1b, tmpbaba)
    wOvXO += _einsum('nb,nMeJ->MbeJ', t1a, tmpabab)
    tmpabab = tmpbaba = None

    # Collect the 1e properties
    Fooa = Fooa.collect()
    Foob = Foob.collect()
    Fvva = Fvva.collect() + F_xv.collect()
    Fvvb = Fvvb.collect() + F_XV.collect()
    u1a = u1a.collect() + u1_ox.collect()
    u1b = u1b.collect() + u1_OX.collect()

    Fova += fova
    Fovb += fovb
    u1a += fova.conj()
    u1a += _einsum('ie,ae->ia', t1a, Fvva)
    u1a -= _einsum('ma,mi->ia', t1a, Fooa)
    u1a -= _einsum('imea,me->ia', t2_ooxv, Fova[:, slc_va]).collect()
    u1a += _einsum('iMaE,ME->ia', t2_oOxV, Fovb).collect()
    u1b += fovb.conj()
    u1b += _einsum('ie,ae->ia', t1b, Fvvb)
    u1b -= _einsum('ma,mi->ia', t1b, Foob)
    u1b -= _einsum('imea,me->ia', t2_OOXV, Fovb[:, slc_vb]).collect()
    u1b += _einsum('mIeA,me->IA', t2_oOxV, Fova[:, slc_va]).collect()

    ooxv, ovxo = map(_get_integral, ('ooxv', 'ovxo'))
    wovxo -= ooxv.transpose(0, 2, 3, 1)
    wovxo += ovxo.transpose(0, 2, 1, 3) # ??
    ooxv -= ovxo.transpose(0, 3, 2, 1)
    u1a -= _einsum('nf,niaf->ia', t1a, ooxv).collect()
    tmp1aa = _einsum('ie,mjbe->mbij', t1a, ooxv).set(label='tmp1aa')
    u2_ooxv += 2. * _einsum('ma,mbij->ijab', t1a, tmp1aa)
    ooxv = ovxo = tmp1aa = None

    OOXV, OVXO = map(_get_integral, ('OOXV', 'OVXO'))
    wOVXO -= OOXV.transpose(0, 2, 3, 1)
    wOVXO += OVXO.transpose(0, 2, 1, 3) # ??
    OOXV -= OVXO.transpose(0, 3, 2, 1)
    u1b -= _einsum('NF,NIaF->IA', t1b, OOXV).collect()
    tmp1bb = _einsum('IE,MJBE->MBIJ', t1b, OOXV).set(label='tmp1bb')
    u2_OOXV += 2. * _einsum('MA,MBIJ->IJAB', t1b, tmp1bb)
    OOXV = OVXO = tmp1bb = None

    ooXV, oxVO = map(_get_integral, ('ooXV', 'oxVO'))
    woVXo -= ooXV.transpose(0, 2, 3, 1)
    woVxO += oxVO.transpose(0, 2, 1, 3)
    u1b += _einsum('nf,nfAI->IA', t1a[:, slc_va], oxVO).collect()
    tmp1ab = _einsum('ie,meBJ->mBiJ', t1a[:, slc_va], oxVO).set(label='tmp1ab')
    tmp1ab += _einsum('IE,mjBE->mBjI', t1b, ooXV)
    tmp1ab = tmp1ab.collect()
    u2_oOxV -= _einsum('ma,mBiJ->iJaB', t1a[:, slc_va], tmp1ab)
    ooXV = oxVO = tmp1ab = None

    OOxv, OVxo = map(_get_integral, ('OOxv', 'OVxo'))
    wOvxO -= OOxv.transpose(0, 2, 3, 1)
    wOvXo += OVxo.transpose(0, 2, 1, 3)
    u1a += _einsum('NF,NFai->ia', t1b, OVxo).collect()
    tmp1ba = _einsum('IE,MEbj->MbIj', t1b, OVxo).set(label='tmp1ba')
    tmp1ba += _einsum('ie,MJbe->MbJi', t1a, OOxv)
    u2_oOxV -= _einsum('MA,MbIj->jIbA', t1b, tmp1ba)
    OOxv = OVxo = tmp1ba = None

    # all the [... slc], ... .collect() are NOT CORRECT
    u2_ooxv += 2. * _einsum('imae,mbej->ijab', t2_ooxv, wovxo)
    u2_ooxv += 2. * _einsum('iMaE,MbEj->ijab', t2_oOxV, wOvXo)
    u2_OOXV += 2. * _einsum('imae,mbej->ijab', t2_OOXV, wOVXO)
    u2_OOXV += 2. * _einsum('mIeA,mBeJ->IJAB', t2_oOxV[:, :, :, slc_vb], woVxO).collect()
    u2_oOxV += 1. * _einsum('imae,mBeJ->iJaB', t2_ooxv, woVxO)
    u2_oOxV += 1. * _einsum('iMaE,MBEJ->iJaB', t2_oOxV, wOVXO)
    u2_oOxV += 1. * _einsum('iMeA,MbeJ->iJbA', t2_oOxV[:, :, :, slc_vb], wOvxO).collect()
    u2_oOxV += 1. * _einsum('IMAE,MbEj->jIbA', t2_OOXV, wOvXo)
    u2_oOxV += 1. * _einsum('mIeA,mbej->jIbA', t2_oOxV, wovxo[:, slc_va]).collect()
    u2_oOxV += 1. * _einsum('mIaE,mBEj->jIaB', t2_oOxV, woVXo)
    wovxo = wOVXO = woVxO = wOvXo = woVXo = wOvxO = None

    Fooa +=  .5 * _einsum('me,ie->mi', fova, t1a)
    Foob +=  .5 * _einsum('me,ie->mi', fovb, t1b)
    Fvva += -.5 * _einsum('me,ma->ae', fova, t1a)
    Fvvb += -.5 * _einsum('me,ma->ae', fovb, t1b)
    Fooa += (eris.focka[:nocca,:nocca] - np.diag(mo_ea_o))
    Foob += (eris.fockb[:noccb,:noccb] - np.diag(mo_eb_o))
    Fvva += (eris.focka[nocca:,nocca:] - np.diag(mo_ea_v))
    Fvvb += (eris.fockb[noccb:,noccb:] - np.diag(mo_eb_v))


    Ftmpa = Fvva - .5 * _einsum('mb,me->be', t1a, Fova)
    Ftmpb = Fvvb - .5 * _einsum('mb,me->be', t1b, Fovb)
    u2_ooxv += _einsum('ijae,be->ijab', t2_ooxv, Ftmpa)
    u2_OOXV += _einsum('ijae,be->ijab', t2_OOXV, Ftmpb)
    u2_oOxV += _einsum('iJaE,BE->iJaB', t2_oOxV, Ftmpb)
    u2_ooxV += _einsum('iJeA,be->iJbA', t2_oOxV, Ftmpa[slc_va])
    Ftmpa = Fooa + .5 * _einsum('je,me->mj', t1a, Fova)
    Ftmpb = Foob + .5 * _einsum('je,me->mj', t1b, Fovb)
    u2_ooxv -= _einsum('imab,mj->ijab', t2_ooxv, Ftmpa)
    u2_OOXV -= _einsum('imab,mj->ijab', t2_OOXV, Ftmpb)
    u2_oOxV -= _einsum('iMaB,MJ->iJaB', t2_oOxV, Ftmpb)
    u2_oOxV -= _einsum('mIaB,mj->jIaB', t2_oOxV, Ftmpa)
    
    oxoo, OXOO, OXoo, oxOO = map(_get_integral, ('oxoo', 'OXOO', 'OXoo', 'oxOO'))
    oxoo = oxoo - oxoo.transpose(2, 1, 0, 3)
    OXOO = OXOO - OXOO.transpose(2, 1, 0, 3)
    u2_ooxv -= _einsum('ma,jbim->ijab', t1a, oxoo)
    u2_OOXV -= _einsum('ma,jbim->ijab', t1b, OXOO)
    u2_oOxV -= _einsum('ma,JBim->iJaB', t1a, OXoo)
    u2_oOxV -= _einsum('MA,ibJM->iJbA', t1b, oxOO)
    oxoo = OXOO = OXoo = oxOO = None

    u2_ooxv *= .5
    u2_OOXV *= .5
    u2_ooxv = u2_ooxv - u2_ooxv.transpose(0,1,3,2)
    u2_ooxv = u2_ooxv - u2_ooxv.transpose(1,0,2,3)
    u2_OOXV = u2_OOXV - u2_OOXV.transpose(0,1,3,2)
    u2_OOXV = u2_OOXV - u2_OOXV.transpose(1,0,2,3)

    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    u1a /= eia_a
    u1b /= eia_b

    u2_ooxv /= lib.direct_sum('ia+jb->ijab', eia_a[:, slc_va], eia_a)
    u2_oOxV /= lib.direct_sum('ia+jb->ijab', eia_a[:, slc_va], eia_b)
    u2_OOXV /= lib.direct_sum('ia+jb->ijab', eia_b[:, slc_vb], eia_b)

    t1new = u1a, u1b
    t2new = u2_ooxv, u2_oOxV, u2_OOXV
    return t1new, t2new


def _init_uccsd(ccsd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import uccsd
    if mpi.rank == 0:
        mpi.comm.bcast((ccsd_obj.mol.dumps(), ccsd_obj.pack()))
    else:
        ccsd_obj = uccsd.UCCSD.__new__(uccsd.UCCSD)
        ccsd_obj.t1 = ccsd_obj.t2 = ccsd_obj.l1 = ccsd_obj.l2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccsd_obj.mol = gto.mole.loads(mol)
        ccsd_obj.unpack_(cc_attr)
    if True:  # If also to initialize cc._scf object
        if mpi.rank == 0:
            if hasattr(ccsd_obj._scf, '_scf'):
                # ZHC FIXME a hack, newton need special treatment to broadcast
                e_tot = ccsd_obj._scf.e_tot
                ccsd_obj._scf = ccsd_obj._scf._scf
                ccsd_obj._scf.e_tot = e_tot
            mpi.comm.bcast((ccsd_obj._scf.__class__, _pack_scf(ccsd_obj._scf)))
        else:
            mf_cls, mf_attr = mpi.comm.bcast(None)
            ccsd_obj._scf = mf_cls(ccsd_obj.mol)
            ccsd_obj._scf.__dict__.update(mf_attr)

    key = id(ccsd_obj)
    mpi._registry[key] = ccsd_obj
    regs = mpi.comm.gather(key)
    return regs


def _init_uiccsd(ccsd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import uccsd
    if mpi.rank == 0:
        mpi.comm.bcast((ccsd_obj.mol.dumps(), ccsd_obj.pack()))
    else:
        ccsd_obj = uccsd.UICCSD.__new__(uccsd.UICCSD)
        ccsd_obj.t1 = ccsd_obj.t2 = ccsd_obj.l1 = ccsd_obj.l2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccsd_obj.mol = gto.mole.loads(mol)
        ccsd_obj.unpack_(cc_attr)
    if True:  # If also to initialize cc._scf object
        if mpi.rank == 0:
            if hasattr(ccsd_obj._scf, '_scf'):
                # ZHC FIXME a hack, newton need special treatment to broadcast
                e_tot = ccsd_obj._scf.e_tot
                ccsd_obj._scf = ccsd_obj._scf._scf
                ccsd_obj._scf.e_tot = e_tot
            mpi.comm.bcast((ccsd_obj._scf.__class__, _pack_scf(ccsd_obj._scf)))
        else:
            mf_cls, mf_attr = mpi.comm.bcast(None)
            ccsd_obj._scf = mf_cls(ccsd_obj.mol)
            ccsd_obj._scf.__dict__.update(mf_attr)

    key = id(ccsd_obj)
    mpi._registry[key] = ccsd_obj
    regs = mpi.comm.gather(key)
    return regs


class UCCSD(pyscf_uccsd.UCCSD):
    """MPI version"""
    conv_tol = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol_normt', 1e-6)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        pyscf_uccsd.UCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        regs = mpi.pool.apply(_init_uccsd, (self, ), (None, ))
        self._reg_procs = regs

    def pack(self):
        packed_args = ('verbose', 'max_memory', 'frozen', 'mo_coeff', 'mo_occ',
                       '_nocc', '_nmo', 'diis_file', 'diis_start_cycle',
                       'level_shift', 'direct', 'diis_space')
        return {arg: getattr(self, arg) for arg in packed_args}

    def unpack_(self, ccdic):
        self.__dict__.update(ccdic)
        return self

    def dump_flags(self, verbose=None):
        if rank == 0:
            pyscf_uccsd.UCCSD.dump_flags(self, verbose)
            logger.info(self, 'level_shift = %.9g', self.level_shift)
            logger.info(self, 'nproc       = %4d', mpi.pool.size)
        return self

    def sanity_check(self):
        if rank == 0:
            pyscf_uccsd.UCCSD.sanity_check(self)
        return self

    init_amps = init_amps


class UICCSD(UCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        pyscf_uccsd.UCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        regs = mpi.pool.apply(_init_uiccsd, (self, ), (None, ))
        self._reg_procs = regs

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        _make_eris_incore_uihf(self, mo_coeff)
        return 'Done'
