"""
MPI Implementation of UCCSD lambda equations

Author: Shuoxue Li <sli7@caltech.edu>
"""
from functools import partial
import numpy as np

from pyscf import lib

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc import uccsd as mpi_uccsd
from mpi4pyscf.cc import cc_tools as tools
from mpi4pyscf.cc.cc_tools import SegArray
from mpi4pyscf.lib import diis
from mpi4pyscf.cc.ccsd import _sync_, _diff_norm

comm = mpi.comm
rank = mpi.rank
ntasks = mpi.pool.size


# everything in intermediates is a SegArray object.
class _IMDS(lib.StreamObject):
    _keys = ('v1a', 'v1b', 'v2a', 'v2b', 'w3a', 'w3b',
            'woooo', 'wOOOO', 'wooOO',
             'wooxo', 'wOOXO', 'wOOxo', 'wooXO',
             'wovxo', 'wOVXO', 'woxVO', 'wOVxo', 'woVXo', 'wOvxO',
             'wvvxo', 'wVVXO', 'wVVxo', 'wvxVO')
    def __init__(self):
        lib.StreamObject.__init__(self)
        for k in self._keys:
            setattr(self, k, None)


def make_intermediates(cc, t1, t2, eris, checkpoint: int | None = None):
    # NOTE [TBO] to be optimized to avoid gather-transpose-scatter

    log = logger.Logger(cc.stdout, cc.verbose)

    debug = (cc.verbose >= logger.DEBUG2)

    t1a, t1b = t1
    t2_ooxv, t2_oOxV, t2_OOXV = t2  # aa, ab, bb
    (nocca, nvira), (noccb, nvirb) = t1a.shape, t1b.shape
    nvira_seg, nvirb_seg = t2_ooxv.shape[2], t2_OOXV.shape[2]
    vlocs_a, vlocs_b = map(tools.get_vlocs, (nvira, nvirb))
    vlocs_ab = (vlocs_a, vlocs_b)
    (vloc0a, vloc1a), (vloc0b, vloc1b) = vlocs_a[rank], vlocs_b[rank]
    slc_va, slc_vb = slice(vloc0a, vloc1a), slice(vloc0b, vloc1b)
    dtype = t1a.dtype

    _fntp = partial(mpi_uccsd.transpose_sz, nvira_or_vlocsa=vlocs_a, nvirb_or_vlocsb=vlocs_b)
    _einsum = partial(mpi_uccsd.einsum_sz, nvira_or_vlocsa=vlocs_a, nvirb_or_vlocsb=vlocs_b)
    _get_integral = partial(mpi_uccsd.get_integral_from_eris, eris=eris, debug=debug)

    fooa = eris.focka[:nocca,:nocca]
    fova = eris.focka[:nocca,nocca:]
    fvoa = eris.focka[nocca:,:nocca]
    fvva = eris.focka[nocca:,nocca:]
    foob = eris.fockb[:noccb,:noccb]
    fovb = eris.fockb[:noccb,noccb:]
    fvob = eris.fockb[noccb:,:noccb]
    fvvb = eris.fockb[noccb:,noccb:]

    tau_ooxv, tau_oOxV, tau_OOXV = mpi_uccsd.make_tau(t2, t1, t1, vlocs=vlocs_ab)

    t2_labels = ('t2aa', 't2ab', 't2bb', 'tau_aa', 'tau_ab', 'tau_bb')
    seg_spins = (0, 0, 1, 0, 0, 1)
    t2_ooxv, t2_oOxV, t2_OOXV, tau_ooxv, tau_oOxV, tau_OOXV = [SegArray(d, 2, s, l, debug=debug)
        for d, s, l in zip((t2_ooxv, t2_oOxV, t2_OOXV, tau_ooxv, tau_oOxV, tau_OOXV),
            seg_spins, t2_labels)]

    labels = ('fooa', 'fova', 'fvoa', 'fvva', 'foob', 'fovb', 'fvob', 'fvvb', 't1a', 't1b')
    fooa, fova, fvoa, fvva, foob, fovb, fvob, fvvb, t1a, t1b = [
        SegArray(data=d, label=l, reduced=True, debug=debug) for d, l in zip(
            (fooa, fova, fvoa, fvva, foob, fovb, fvob, fvvb, t1a, t1b), labels
        )
    ]

    t1_ox = t1a[:, slc_va].set(label='t1_ox', seg_spin=0)
    t1_OX = t1b[:, slc_vb].set(label='t1_OX', seg_spin=1)

    oxov, OXOV, oxOV = map(_get_integral, ("oxov", "OXOV", "oxOV"))
    oxov = _fntp(oxov, 1, 3, 1.0, -1.0)
    OXOV = _fntp(OXOV, 1, 3, 1.0, -1.0)

    v1a_0  = fvva - _einsum('ja,jb->ba', fova, t1a)
    v1b_0  = fvvb - _einsum('ja,jb->ba', fovb, t1b)
    v2a_0  = fooa + _einsum('ib,jb->ij', fova, t1a)
    v2b_0  = foob + _einsum('ib,jb->ij', fovb, t1b)

    v1a_0 = SegArray(data=v1a_0, label='v1a', debug=debug, reduced=True)
    v1b_0 = SegArray(data=v1b_0, label='v1b', debug=debug, reduced=True)
    v2a_0 = SegArray(data=v2a_0, label='v2a', debug=debug, reduced=True)
    v2b_0 = SegArray(data=v2b_0, label='v2b', debug=debug, reduced=True)

    v1_xv = SegArray(np.zeros((nvira_seg, nvira), dtype=dtype), seg_idx=0, seg_spin=0, label='v1a_xv', debug=debug)
    v1_XV = SegArray(np.zeros((nvirb_seg, nvirb), dtype=dtype), seg_idx=0, seg_spin=1, label='v1b_XV', debug=debug)

    v1_xv += _einsum('jcka,jkbc->ba', oxov, tau_ooxv) * .5 # [c],[b]c->[b]

    v1_xv -= _einsum('jaKC,jKbC->ba', oxOV, tau_oOxV, out=v1_xv) * .5    # [a],[b]->[b]a
    v1_xv -= _einsum('kaJC,kJbC->ba', oxOV, tau_oOxV, out=v1_xv) * .5    # [a],[b]->[b]a
    v1_XV += _einsum('jcka,jkbc->ba', OXOV, tau_OOXV) * .5    # [c],[b]c->[b]
    v1b = -.5 * _einsum('kcJA,kJcB->BA', oxOV, tau_oOxV).set(label='v1b') # [c],[c]->...
    v1b -= _einsum('jcKA,jKcB->BA', oxOV, tau_oOxV) * .5    # [c],[c]->...
    v2a  = .5 * _einsum('ibkc,jkbc->ij', oxov, tau_ooxv).set(label='v2a')    # [b],[b]->...    
    v2a += _einsum('ibKC,jKbC->ij', oxOV, tau_oOxV)
    v2b  = .5 * _einsum('ibkc,jkbc->ij', OXOV, tau_OOXV).set(label='v2b')
    v2b += _einsum('kcIB,kJcB->IJ', oxOV, tau_oOxV)

    oxoo, OXOO, OXoo, oxOO = map(_get_integral, ("oxoo", "OXOO", "OXoo", "oxOO"))
    oxoo = _fntp(oxoo, 0, 2, 1.0, -1.0)
    OXOO = _fntp(OXOO, 0, 2, 1.0, -1.0)

    v2a -= _einsum('ibkj,kb->ij', oxoo, t1_ox)
    v2a += _einsum('KBij,KB->ij', OXoo, t1_OX)
    v2b -= _einsum('ibkj,kb->ij', OXOO, t1_OX)
    v2b += _einsum('kbIJ,kb->IJ', oxOO, t1_ox)

    v5a  = fvoa + _einsum('kc,jkbc->bj', fova, t2_ooxv)     # ...,[b]->[b]
    v5a += _einsum('KC,jKbC->bj', fovb, t2_oOxV)            # ...,[b]->[b]
    v5b  = fvob + _einsum('kc,jkbc->bj', fovb, t2_OOXV)     # 
    v5b += _einsum('kc,kJcB->BJ', fova[:, slc_va], t2_oOxV)
    tmp  = fova - _einsum('kdlc,ld->kc', oxov, t1a[:, slc_va])
    tmp += _einsum('kcLD,LD->kc', oxOV, t1b)    # [c],...,->[c]

    v5a += np.einsum('kc,kb,jc->bj', tmp.data, t1a.data, t1a.data)
    tmp  = fovb - _einsum('kdlc,ld->kc', OXOV, t1_OX)
    tmp += _einsum('ldKC,ld->KC', oxOV, t1_ox)
    v5b += np.einsum('kc,kb,jc->bj', tmp.data, t1b.data, t1b.data)
    v5a -= _einsum('lckj,klbc->bj', oxoo, t2_ooxv) * .5 # [c],[b]c->[b]
    v5a -= _einsum('LCkj,kLbC->bj', OXoo, t2_oOxV)      # [C],[b]C->[b]
    v5b -= _einsum('LCKJ,KLBC->BJ', OXOO, t2_OOXV) * .5 # [C],[B]C->[B]
    v5b -= _einsum('lcKJ,lKcB->BJ', oxOO, t2_oOxV)      # [c],[c]->...

    oooo, OOOO, ooOO = map(_get_integral, ("oooo", "OOOO", "ooOO"))
    woooo  = _einsum('icjl,kc->ikjl', oxoo, t1_ox)
    wOOOO  = _einsum('icjl,kc->ikjl', OXOO, t1_OX)
    wooOO  = _einsum('icJL,kc->ikJL', oxOO, t1_ox)
    wooOO += _einsum('JCil,KC->ilJK', OXoo, t1_OX)

    woooo = woooo.collect()
    wOOOO = wOOOO.collect()
    wooOO = wooOO.collect()

    # FIXME Why I cannot just use oooo (SegArray obj)?
    woooo += (oooo.data - oooo.data.transpose(0, 3, 2, 1)) * .5
    wOOOO += (OOOO.data - OOOO.data.transpose(0, 3, 2, 1)) * .5
    wooOO += ooOO.data.conj()

    if checkpoint == 4:
        return dict(woooo=woooo.data, wOOOO=wOOOO.data, wooOO=wooOO.data,
                    oooo=oooo.data, OOOO=OOOO.data)

    woooo += _einsum('icjd,klcd->ikjl', oxov, tau_ooxv) * .25
    wOOOO += _einsum('icjd,klcd->ikjl', OXOV, tau_OOXV) * .25
    wooOO += _einsum('icJD,kLcD->ikJL', oxOV, tau_oOxV)


    if checkpoint == 5:
        return dict(ovoo=oxoo.collect().data,
                    OVOO=OXOO.collect().data,
                    ovOO=oxOO.collect().data,
                    OVoo=OXoo.collect().data,
                    ovov=oxov.collect().data,
                    OVOV=OXOV.collect().data,
                    ovOV=oxOV.collect().data,
                    tauaa=tau_ooxv.collect().data,
                    taubb=tau_OOXV.collect().data,
                    tauab=tau_oOxV.collect().data,
                    woooo=woooo.data,
                    wOOOO=wOOOO.data,
                    wooOO=wooOO.data,
                    )

    ovxo, ooxv, OVXO, OOXV, OVxo, oxVO, ooXV, OOxv = map(_get_integral,
        ("ovxo", "ooxv", "OVXO", "OOXV", "OVxo", "oxVO", "ooXV", "OOxv"))

    v4ovxo = SegArray(data=np.zeros((nocca, nvira, nvira_seg, nocca), dtype=dtype), seg_idx=2, seg_spin=0, label='v4ovvo', debug=debug)
    v4OVXO = SegArray(data=np.zeros((noccb, nvirb, nvirb_seg, noccb), dtype=dtype), seg_idx=2, seg_spin=1, label='v4OVVO', debug=debug)
    v4oxVO = SegArray(data=np.zeros((nocca, nvira_seg, nvirb, noccb), dtype=dtype), seg_idx=1, seg_spin=0, label='v4ovVO', debug=debug)
    v4OVxo = SegArray(data=np.zeros((noccb, nvirb, nvira_seg, nocca), dtype=dtype), seg_idx=2, seg_spin=0, label='v4oVvo', debug=debug)
    v4oVXo = SegArray(data=np.zeros((nocca, nvirb, nvirb_seg, nocca), dtype=dtype), seg_idx=2, seg_spin=1, label='v4oVVo', debug=debug)
    v4OvxO = SegArray(data=np.zeros((noccb, nvira, nvira_seg, noccb), dtype=dtype), seg_idx=2, seg_spin=0, label='v4OvvO', debug=debug)

    # PySCF L114-L131
    v4ovxo += _einsum('jbld,klcd->jbck', oxov, t2_ooxv)
    v4ovxo += _einsum('jbLD,kLcD->jbck', oxOV, t2_oOxV) # [b],[c]->b[c]
    v4ovxo += ovxo
    v4ovxo -= ooxv.transpose(0,3,2,1)   # NOTE [TBO]
    v4OVXO += _einsum('jbld,klcd->jbck', OXOV, t2_OOXV)
    v4OVXO += _einsum('ldJB,lKdC->JBCK', oxOV, t2_oOxV, out=v4OVXO)   # [d],[d]C->[C]
    v4OVXO += OVXO  
    v4OVXO -= OOXV.transpose(0,3,2,1)   # NOTE [TBO]
    v4OVxo += _einsum('ldJB,klcd->JBck', oxOV, t2_ooxv, out=v4OVxo) # [d],[c]d->[c]
    v4OVxo += _einsum('JBLD,kLcD->JBck', OXOV, t2_oOxV, out=v4OVxo) # [B],[c]->B[c]
    v4OVxo += OVxo
    v4oxVO += _einsum('jbLD,KLCD->jbCK', oxOV, t2_OOXV, out=v4oxVO) # [b],[C]->[b]C
    v4oxVO += _einsum('jbld,lKdC->jbCK', oxov, t2_oOxV) # [b]d,[d]->[b]
    v4oxVO += oxVO
    v4oVXo += _einsum('jdLB,kLdC->jBCk', oxOV, t2_oOxV, out=v4oVXo) # [d],[d]C->[C]
    v4oVXo -= ooXV.transpose(0, 3, 2, 1)
    v4OvxO += _einsum('lbJD,lKcD->JbcK', oxOV, t2_oOxV, out=v4OvxO) # [b],[c]->b[c]
    v4OvxO -= OOxv.transpose(0, 3, 2, 1)


    wooxo  = _einsum('ibck,jb->ijck', v4ovxo, t1a).set(label='woovo', debug=debug)
    wOOXO  = _einsum('ibck,jb->ijck', v4OVXO, t1b).set(label='wOOVO', debug=debug)
    wOOxo  = _einsum('IBck,JB->IJck', v4OVxo, t1b).set(label='wOOvo', debug=debug)
    wOOxo -= _einsum('IbcK,jb->IKcj', v4OvxO, t1a)
    wooXO = SegArray(np.zeros((nocca, nocca, nvirb_seg, noccb), dtype=dtype),
                     seg_idx=2, seg_spin=1, label='wooVO', debug=debug)
    wooXO += _einsum('ibCK,jb->ijCK', v4oxVO, t1_ox, out=wooXO)   # [b],[b]->...
    wooXO -= _einsum('iBCk,JB->ikCJ', v4oVXo, t1b)  # [C],...->[C]
    wooxo += oxoo.conj().transpose(3, 2, 1, 0) * .5
    wOOXO += OXOO.conj().transpose(3, 2, 1, 0) * .5
    wooXO += OXoo.conj().transpose(3, 2, 1, 0)
    wOOxo += oxOO.conj().transpose(3, 2, 1, 0)
    wooxo -= _einsum('iclk,jlbc->ikbj', oxoo, t2_ooxv)
    wooxo += _einsum('LCik,jLbC->ikbj', OXoo, t2_oOxV)
    wOOXO -= _einsum('iclk,jlbc->ikbj', OXOO, t2_OOXV)
    wOOXO += _einsum('lcIK,lJcB->IKBJ', oxOO, t2_oOxV, out=wOOXO)  # [c],[c]B->[B]
    wooXO -= _einsum('iclk,lJcB->ikBJ', oxoo, t2_oOxV, out=wooXO)  # [c],[c]B->[B]
    wooXO += _einsum('LCik,JLBC->ikBJ', OXoo, t2_OOXV)
    wooXO -= _einsum('icLK,jLcB->ijBK', oxOO, t2_oOxV)
    wOOxo -= _einsum('ICLK,jLbC->IKbj', OXOO, t2_oOxV)
    wOOxo += _einsum('lcIK,jlbc->IKbj', oxOO, t2_ooxv)
    wOOxo -= _einsum('IClk,lJbC->IJbk', OXoo, t2_oOxV)

    wvvxo  = _einsum('jack,jb->back', v4ovxo, t1a)
    wVVXO  = _einsum('jack,jb->back', v4OVXO, t1b)
    wVVxo  = _einsum('JAck,JB->BAck', v4OVxo, t1b)
    wVVxo -= _einsum('jACk,jb->CAbk', v4oVXo, t1_ox, out=wVVxo) # [C],[b]->C[b]
    wvxVO  = _einsum('jaCK,jb->baCK', v4oxVO, t1a)
    wvxVO -= _einsum('JacK,JB->caBK', v4OvxO, t1b)

    # wvvxo += _einsum('lajk,jlbc->back', .25*oxoo, tau_ooxv)   # use anti-symmetry
    wvvxo -= _einsum('lajk,jlcb->back', .25 * oxoo, tau_ooxv)

    # wVVVO += _einsum('lajk,jlbc->back', .25*OXOO, tau_OOXV)
    wVVXO -= _einsum('lajk,jlcb->back', .25 * OXOO, tau_OOXV)

    wVVxo -= _einsum('LAjk,jLcB->BAck', OXoo, tau_oOxV)
    wvxVO -= _einsum('laJK,lJbC->baCK', oxOO, tau_oOxV) # [a],[b]->b[a]

    w3a  = _einsum('jbck,jb->ck', v4ovxo, t1a)
    w3a += _einsum('JBck,JB->ck', v4OVxo, t1b)
    w3a = w3a.collect()
    w3b  = _einsum('jbck,jb->ck', v4OVXO, t1b).collect()
    w3b += _einsum('jbCK,jb->CK', v4oxVO, t1_ox).collect()

    # FIXME: current v4OVvo(v4OVxo) has larger error

    if checkpoint == 10:
        return dict(
                    v1a=(v1a_0 + v1_xv.collect()).data,
                    v1b=(v1b_0 + v1b.collect() + v1_XV.collect()).data,
                    v4OVvo=v4OVxo.collect().data,
                    v4ovvo=v4ovxo.collect().data,
                    wVVvo=wVVxo.collect().data,
                    OVvo=OVxo.collect().data,
                    w3a=w3a.data,
                    w3b=w3b.data,
                    )

    # pyscf L170-L187
    # v4ovxo += _einsum('jbld,kd,lc->jbck', oxov, t1a, -t1a)
    tmp = _einsum("jbld,kd->jblk", oxov, t1a)
    print(f"tmp.seg_spin = {tmp.seg_spin} oxov.seg_spin = {oxov.seg_spin}")
    v4ovxo += _einsum('jblk,lc->jbck', tmp, -t1_ox.set(seg_spin=0), out=v4ovxo)
    # v4OVXO += _einsum('jbld,kd,lc->jbck', OXOV, t1b, -t1b)
    v4OVXO += _einsum("jblk,lc->jbck", _einsum("jbld,kd->jblk", OXOV, t1b), -t1_OX)
    # v4oxVO += _einsum('jbLD,KD,LC->jbCK', oxOV, t1b, -t1b)
    v4oxVO += _einsum("jbLK,LC->jbCK", _einsum("jbLD,KD->jbLK", oxOV, t1b), -t1_OX)
    # v4OVxo += _einsum('ldJB,kd,lc->JBck', oxOV, t1a, -t1a)
    tmp = SegArray(np.zeros((nvira, nvira_seg, noccb, nvirb)), seg_idx=1, seg_spin=0, label='tmp', debug=debug)
    tmp += _einsum("ldJB,lc->dcJB", oxOV, -t1_ox, out=tmp)
    v4OVxo += _einsum("dcJB,kd->JBck", tmp, t1a)
    # v4oVXo += _einsum('jdLB,kd,LC->jBCk', oxOV, t1a, t1b)

    tmp = SegArray(np.zeros((nocca, nvira, nvirb, nvirb_seg)), seg_idx=3, seg_spin=1, label='tmp', debug=debug)
    tmp += _einsum("jdLB,LC->jdBC", oxOV, t1_OX, out=tmp)
    v4oVXo += _einsum("jdBC,kd->jBCk", tmp, t1a)

    # v4OvxO += _einsum('lbJD,KD,lc->JbcK', oxOV, t1b, t1a)
    tmp = SegArray(np.zeros((nvira, nvira_seg, noccb, nvirb)), seg_idx=1, seg_spin=0, label='tmp', debug=debug)
    tmp += _einsum("lbJD,lc->bcJD", oxOV, t1_ox, out=tmp)
    v4OvxO += _einsum("bcJD,KD->JbcK", tmp, t1b)

    tmp = None

    v4ovxo -= _einsum('jblk,lc->jbck', oxoo, t1a)
    v4OVXO -= _einsum('jblk,lc->jbck', OXOO, t1b)
    v4oxVO -= _einsum('jbLK,LC->jbCK', oxOO, t1b)
    v4OVxo -= _einsum('JBlk,lc->JBck', OXoo, t1a)
    v4oVXo += _einsum('LBjk,LC->jBCk', OXoo, t1b)
    v4OvxO += _einsum('lbJK,lc->JbcK', oxOO, t1a)

    wovxo, wOVXO, woxVO, wOVxo, woVXo, wOvxO = v4ovxo, v4OVXO, v4oxVO, v4OVxo, v4oVXo, v4OvxO

    # v1a = v1a_0 + v1_xv.collect()
    # v1b = v1b_0 + v1b.collect() + v1_XV.collect()
    v2a = v2a_0 + v2a.collect()
    v2b = v2b_0 + v2b.collect()

    if checkpoint == 15:
        v1_xv = v1_xv.collect()
        v1b = v1b.collect()
        v1_XV = v1_XV.collect()
        v1a = v1a_0 + v1_xv
        v1b = v1b_0 + v1b + v1_XV
        return dict(v1a=v1a.data, v1b=v1b.data)

    if nvira > 0 and nocca > 0:
        oxvv = _get_integral("oxvv")
        oxvv = _fntp(oxvv, 1, 3, 1.0, -1.0)
        # v1a -= _einsum('jabc,jc->ba', oxvv, t1a).collect()
        v1_xv -= _einsum('jabc,jc->ba', oxvv, t1a)  # err=1e-12
        # v1_xv += _einsum("jabc,jc->ba", oxvv, t1a).transpose(1, 0)    # err=1e-7

        v5a += _einsum('kdbc,jkcd->bj', oxvv, t2_ooxv) * .5     # [d]c,[c]d->...
        wooxo += _einsum('idcb,kjbd->ijck', oxvv, tau_ooxv) * .25   # [d]b,[b]d->[c] ???
        wovxo += _einsum('jbcd,kd->jbck', oxvv, t1a)    # [b]c,...->[c]

        wvvxo -= oxvv.conj().transpose(3,2,1,0) * .5
        wvvxo += _einsum('jacd,kjbd->cabk', oxvv, t2_ooxv)
        wvxVO += _einsum('jacd,jKdB->caBK', oxvv, t2_oOxV)  # [a]d,[d]->[a]
        oxvv = None

    if nvirb > 0 and noccb > 0:
        OXVV = _get_integral("OXVV")
        OXVV = _fntp(OXVV, 1, 3, 1.0, -1.0)
        
        v1_XV -= _einsum('jabc,jc->ba', OXVV, t1b)
        # v1_XV += _einsum('jabc,jc->ba', OXVV, t1b).transpose(1, 0)
        # comm.Barrier()
        v5b += _einsum('KDBC,JKCD->BJ', OXVV, t2_OOXV) * .5
        wOOXO += _einsum('idcb,kjbd->ijck', OXVV, tau_OOXV) * .25    
        wOVXO += _einsum('jbcd,kd->jbck', OXVV, t1b)
        wVVXO -= OXVV.conj().transpose(3,2,1,0) * .5
        wVVXO += _einsum('jacd,kjbd->cabk', OXVV, t2_OOXV)
        wVVxo += _einsum('JACD,kJbD->CAbk', OXVV, t2_oOxV)
        if checkpoint == 18:
            return dict(
                OVVV=OXVV.collect().data,
                t1a=t1a.data, t1b=t1b.data,
                v1a=(v1a_0 + v1_xv.collect()).data,
                v1b=(v1b_0 + v1b.collect() + v1_XV.collect()).data,
                )

        OXVV = None


    if nvirb > 0 and nocca > 0:
        OXvv = _get_integral("OXvv")
        v1a = _einsum('JCba,JC->ba', OXvv, t1_OX)
        v5a += _einsum('KDbc,jKcD->bj', OXvv, t2_oOxV)
        wOOxo += _einsum('IDcb,kJbD->IJck', OXvv, tau_oOxV)
        wOVxo += _einsum('JBcd,kd->JBck', OXvv, t1a)    # [B],...->[B]
        wOvxO -= _einsum('JDcb,KD->JbcK', OXvv, t1_OX)
        wvxVO -= OXvv.conj().transpose(3,2,1,0)
        wvvxo -= _einsum('KDca,jKbD->cabj', OXvv, t2_oOxV)  # [D],[b]D->[b]
        wvvXO = - _einsum('KDca,JKBD->caBJ', OXvv, t2_OOXV) # [D],[B]D->[B]
        wvxVO += wvvXO
        wVVxo += _einsum('KAcd,jKdB->BAcj', OXvv, t2_oOxV)
        if checkpoint == 19:
            return dict(
                w3a=w3a.data, w3b=w3b.data,
                OVvv=OXvv.collect().data)
        OXvv = tmp = None

    if nvira > 0 and noccb > 0:
        oxVV = _get_integral("oxVV")
        v1b += _einsum('jcBA,jc->BA', oxVV, t1_ox)
        v5b += _einsum('kdBC,kJdC->BJ', oxVV, t2_oOxV)
        wooXO += _einsum('idCB,jKdB->ijCK', oxVV, tau_oOxV)
        woxVO += _einsum('jbCD,KD->jbCK', oxVV, t1b)
        woVXo -= _einsum('jdCB,kd->jBCk', oxVV, t1_ox)
        wVVxo -= oxVV.conj().transpose(3,2,1,0)
        wVVXO -= _einsum('kdCA,kJdB->CABJ', oxVV, t2_oOxV, out=wVVXO)   # [d],[d]B->B
        wVVxo -= _einsum('kdCA,jkbd->CAbj', oxVV, t2_ooxv, out=wVVxo)
        wvxVO += _einsum('kaCD,kJbD->baCJ', oxVV, t2_oOxV, out=wvxVO)   # [a],[b]->b[a]
        oxVV = tmp = None

    w3a += v5a.collect()
    w3b += v5b.collect()

    v1_xv = v1_xv.collect()
    v1a = v1a.collect()
    v1b = v1b.collect()
    v1_XV = v1_XV.collect()
    v1a = v1a_0 + v1a + v1_xv
    v1b = v1b_0 + v1b + v1_XV

    w3a += lib.einsum('cb,jb->cj', v1a.data, t1a.data)
    w3b += lib.einsum('cb,jb->cj', v1b.data, t1b.data)
    w3a -= lib.einsum('jk,jb->bk', v2a.data, t1a.data)
    w3b -= lib.einsum('jk,jb->bk', v2b.data, t1b.data)

    if checkpoint == 20:    # w3b
        return dict(
            v1a=v1a.data, v1b=v1b.data,
            v2a=v2a.data, v2b=v2b.data,
            w3a=w3a.data, w3b=w3b.data,
            v5a=v5a.data, v5b=v5b.data
        )

    imds = _IMDS().set(
        woooo=woooo, wOOOO=wOOOO, wooOO=wooOO,
        wooxo=wooxo, wOOXO=wOOXO, wOOxo=wOOxo, wooXO=wooXO,
        wovxo=wovxo, wOVXO=wOVXO, woxVO=woxVO, wOVxo=wOVxo, woVXo=woVXo, wOvxO=wOvxO,
        wvvxo=wvvxo, wVVXO=wVVXO, wVVxo=wVVxo, wvxVO=wvxVO,
        v1a=v1a, v1b=v1b, v2a=v2a, v2b=v2b, w3a=w3a, w3b=w3b,
    )
    return imds


def update_lambda(cc, t1, t2, l1, l2, eris, imds, checkpoint: int | None = None,
                  return_segarray: bool = False):
    log = logger.Logger(cc.stdout, cc.verbose)

    debug = (cc.verbose >= logger.DEBUG2)

    t1a, t1b = t1
    t2_ooxv, t2_oOxV, t2_OOXV = t2  # aa, ab, bb
    l1a, l1b = l1
    l2_ooxv, l2_oOxV, l2_OOXV = l2
    (nocca, nvira), (noccb, nvirb) = t1a.shape, t1b.shape
    nvira_seg, nvirb_seg = t2_ooxv.shape[2], t2_OOXV.shape[2]
    vlocs_a, vlocs_b = map(tools.get_vlocs, (nvira, nvirb))
    vlocs_ab = (vlocs_a, vlocs_b)
    (vloc0a, vloc1a), (vloc0b, vloc1b) = vlocs_a[rank], vlocs_b[rank]
    slc_va, slc_vb = slice(vloc0a, vloc1a), slice(vloc0b, vloc1b)
    dtype = t1a.dtype

    _fntp = partial(mpi_uccsd.transpose_sz, nvira_or_vlocsa=vlocs_a, nvirb_or_vlocsb=vlocs_b)
    _einsum = partial(mpi_uccsd.einsum_sz, nvira_or_vlocsa=vlocs_a, nvirb_or_vlocsb=vlocs_b)
    _get_integral = partial(mpi_uccsd.get_integral_from_eris, eris=eris, debug=debug)

    u1a, u1b = np.zeros_like(t1a), np.zeros_like(t1b)
    u1a0, u1b0 = np.zeros_like(t1a), np.zeros_like(t1b)
    u2_ooxv = np.zeros((nocca, nocca, nvira_seg, nvira), dtype=dtype)
    u2_oOxV = np.zeros((nocca, noccb, nvira_seg, nvirb), dtype=dtype)
    u2_OOXV = np.zeros((noccb, noccb, nvirb_seg, nvirb), dtype=dtype)
    tau_ooxv, tau_oOxV, tau_OOXV = mpi_uccsd.make_tau(t2, t1, t1)
    m3aa, m3ab, m3bb = mpi_uccsd._add_vvvv(t1=None, t2=(l2_ooxv.conj(),l2_oOxV.conj(),l2_OOXV.conj()), eris=eris, vlocs=vlocs_ab)
    m3aa = m3aa.conj()
    m3ab = m3ab.conj()
    m3bb = m3bb.conj()

    # Make SegArray objects
    seg_spins = [0, 0, 1] * 5
    labels = ('t2aa', 't2ab', 't2bb',
              'l2aa', 'l2ab', 'l2bb',
              'u2aa', 'u2ab', 'u2bb',
              'm3aa', 'm3ab', 'm3bb',
              'tauaa', 'tauab', 'taubb')
    arrs = (t2_ooxv, t2_oOxV, t2_OOXV,
             l2_ooxv, l2_oOxV, l2_OOXV,
             u2_ooxv, u2_oOxV, u2_OOXV,
             m3aa, m3ab, m3bb,
             tau_ooxv, tau_oOxV, tau_OOXV)
    t2_ooxv, t2_oOxV, t2_OOXV, l2_ooxv, l2_oOxV, l2_OOXV, u2_ooxv, u2_oOxV, u2_OOXV, m3aa, m3ab, m3bb, tau_ooxv, tau_oOxV, tau_OOXV = [
        SegArray(arr, 2, seg_spin, label, debug=debug) for arr, seg_spin, label in zip(arrs, seg_spins, labels)]

    if checkpoint == 4:
        return dict(
            t2aa=t2_ooxv.collect().data,
            t2bb=t2_OOXV.collect().data,
            t2ab=t2_oOxV.collect().data,
            l2aa=l2_ooxv.collect().data,
            l2bb=l2_OOXV.collect().data,
            l2ab=l2_oOxV.collect().data,
            m3aa=m3aa.collect().data,
            m3bb=m3bb.collect().data,
            m3ab=m3ab.collect().data)

    labels = ('t1a', 't1b', 'l1a', 'l1b', 'u1a0', 'u1b0', 'u1a', 'u1b')
    reduced = (True, True, True, True, True, True, False, False)
    t1a, t1b, l1a, l1b, u1a0, u1b0, u1a, u1b = [SegArray(arr, None, None, label, debug=debug, reduced=reduced)
                                    for arr, label in zip((t1a, t1b, l1a, l1b, u1a0, u1b0, u1a, u1b), labels)]

    t1_ox, t1_OX = t1a[:, slc_va].set(label='t1_ox'), t1b[:, slc_vb].set(label='t1_OX')
    l1_ox, l1_OX = l1a[:, slc_va].set(label='l1_ox'), l1b[:, slc_vb].set(label='l1_OX')
    u1_ox, u1_OX = u1a[:, slc_va].set(label='u1_ox'), u1b[:, slc_vb].set(label='u1_OX')

    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + cc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + cc.level_shift

    fova = eris.focka[:nocca, nocca:]
    fovb = eris.fockb[:noccb, noccb:]
    v1a = imds.v1a - np.diag(mo_ea_v)
    v1b = imds.v1b - np.diag(mo_eb_v)
    v2a = imds.v2a - np.diag(mo_ea_o)
    v2b = imds.v2b - np.diag(mo_eb_o)

    mvv  = _einsum('klca,klcb->ba', l2_ooxv, t2_ooxv).collect().set(label='mvv') * .5
    mvv += _einsum('lKaC,lKbC->ba', l2_oOxV, t2_oOxV).collect() # [a],[b]->ab
    mVV  = _einsum('klca,klcb->ba', l2_OOXV, t2_OOXV).set(label='mVV').collect() * .5   # [c],[c]->...
    mVV += _einsum('kLcA,kLcB->BA', l2_oOxV, t2_oOxV).collect()
    moo  = _einsum('kicd,kjcd->ij', l2_ooxv, t2_ooxv).set(label='moo') * .5 # [c],[c]->...
    moo += _einsum('iKdC,jKdC->ij', l2_oOxV, t2_oOxV)   # [d],[d]->...
    moo = moo.collect()
    mOO  = _einsum('kicd,kjcd->ij', l2_OOXV, t2_OOXV).set(label='mOO') * .5 # [c],[c]->...
    mOO += _einsum('kIcD,kJcD->IJ', l2_oOxV, t2_oOxV)   # [c],[c]->...
    mOO = mOO.collect()

    m3aa += _einsum('klab,ikjl->ijab', l2_ooxv, imds.woooo)
    m3bb += _einsum('klab,ikjl->ijab', l2_OOXV, imds.wOOOO)
    m3ab += _einsum('kLaB,ikJL->iJaB', l2_oOxV, imds.wooOO)

    if checkpoint == 5:
        return dict(
            mvv=mvv.data, mVV=mVV.data,
            moo=moo.data, mOO=mOO.data,
            m3aa=m3aa.collect().data,
            m3bb=m3bb.collect().data,
            m3ab=m3ab.collect().data,
            v2a=v2a.data)

    oxov, OXOV, oxOV = map(_get_integral, ("oxov", "OXOV", "oxOV"))
    oxov = _fntp(oxov, 1, 3, 1.0, -1.0)
    OXOV = _fntp(OXOV, 1, 3, 1.0, -1.0)

    mvv1 = mvv + lib.einsum('jc,jb->bc', l1a.data, t1a.data)
    mVV1 = mVV + lib.einsum('jc,jb->bc', l1b.data, t1b.data)
    moo1 = moo + lib.einsum('ic,kc->ik', l1a.data, t1a.data)
    mOO1 = mOO + lib.einsum('ic,kc->ik', l1b.data, t1b.data)

    if nvira > 0 and nocca > 0:
        oxvv = _get_integral("oxvv")
        oxvv = _fntp(oxvv, 1, 3, 1.0, -1.0)
        tmp = _einsum('ijcd,kd->ijck', l2_ooxv, t1a)
        m3aa -= _einsum('kbca,ijck->ijab', oxvv, tmp) # [b]c,[c]->[b]

        tmp = _einsum('ic,jbca->jiba', l1a, oxvv) # ...,[b]->[b]  seg=2
        tmp += _einsum('kiab,jk->ijab', l2_ooxv, v2a)
        tmp -= _einsum('ik,kajb->ijab', moo1, oxov)
        u2_ooxv += tmp - tmp.transpose(1, 0, 2, 3)
        u1_ox += _einsum('iacb,bc->ia', oxvv, mvv1)   # [a],...->[a]
        oxvv = tmp = None

    if nvirb > 0 and noccb > 0:
        OXVV = _get_integral("OXVV")
        OXVV = OXVV - OXVV.transpose(0,3,2,1)
        tmp = _einsum('ijcd,kd->ijck', l2_OOXV, t1b)
        m3bb -= _einsum('kbca,ijck->ijab', OXVV, tmp)

        tmp = _einsum('ic,jbca->jiba', l1b, OXVV)
        tmp += _einsum('kiab,jk->ijab', l2_OOXV, v2b)
        tmp -= _einsum('ik,kajb->ijab', mOO1, OXOV)
        u2_OOXV += tmp - tmp.transpose(1, 0, 2, 3)
        u1_OX += _einsum('iaCB,BC->ia', OXVV, mVV1)   # [a],...->[a]
        OXVV = tmp = None

    if nvirb > 0 and nocca > 0:
        OXvv = _get_integral("OXvv")
        tmp = _einsum('iJcD,KD->iJcK', l2_oOxV, t1b)
        m3ab -= _einsum('KBca,iJcK->iJaB', OXvv, tmp)
        tmp = SegArray(np.zeros((noccb, nocca, nvira_seg, nvirb)), seg_idx=2, seg_spin=0, label='tmp', debug=debug)
        tmp -= _einsum('kIaB,jk->IjaB', l2_oOxV, v2a)
        tmp -= _einsum('IK,jaKB->IjaB', mOO1, oxOV)
        tmp += _einsum('ic,JAcb->JibA', l1a, OXvv)
        u2_oOxV += tmp.transpose(1, 0, 2, 3)
        u1_OX += _einsum('iacb,bc->ia', OXvv, mvv1)
        OXvv = tmp = None


    if nvira > 0 and noccb > 0:
        oxVV = _get_integral("oxVV")
        tmp = SegArray(np.zeros((nocca, noccb, nvirb_seg, nocca)), seg_idx=2, seg_spin=1, label='tmp', debug=debug)
        tmp += _einsum('iJdC,kd->iJCk', l2_oOxV, t1_ox, out=tmp)    # [d]C,[d]->[C]
        m3ab -= _einsum('kaCB,iJCk->iJaB', oxVV, tmp, out=m3ab)     # [a]C,C->[a]
        tmp = _einsum('IC,jbCA->jIbA', l1b, oxVV).set(label='tmp', debug=debug)
        tmp -= _einsum('iKaB,JK->iJaB', l2_oOxV, v2b)
        tmp -= _einsum('ik,kaJB->iJaB', moo1, oxOV)
        u2_oOxV += tmp
        u1_ox += _einsum('iaCB,BC->ia', oxVV, mVV1)
        oxVV = tmp = None

    if checkpoint == 10:
        return dict(
            u2aa=u2_ooxv.collect().data,
            u2bb=u2_OOXV.collect().data,
            u2ab=u2_oOxV.collect().data,
            u1a = u1a0.data + u1_ox.collect().data + u1a.collect().data,
            u1b = u1b0.data + u1_OX.collect().data + u1b.collect().data,
        )

    oxov, OXOV, oxOV = map(_get_integral, ("oxov", "OXOV", "oxOV"))

    tmp = _einsum('ijcd,klcd->ijkl', l2_ooxv, tau_ooxv).collect().set(label='tmp', debug=debug)
    oxov = _fntp(oxov, 1, 3, 1.0, -1.0)
    m3aa += _einsum('kalb,ijkl->ijab', oxov, tmp) * .25
 
    tmp = _einsum('ijcd,klcd->ijkl', l2_OOXV, tau_OOXV).collect().set(label='tmp', debug=debug)
    OXOV = _fntp(OXOV, 1, 3, 1.0, -1.0)
    m3bb += _einsum('kalb,ijkl->ijab', OXOV, tmp) * .25

    tmp = _einsum('iJcD,kLcD->iJkL', l2_oOxV, tau_oOxV).collect().set(label='tmp', debug=debug)
    m3ab += _einsum('kaLB,iJkL->iJaB', oxOV, tmp) * .5
    tmp = _einsum('iJdC,lKdC->iJKl', l2_oOxV, tau_oOxV).collect().set(label='tmp', debug=debug)
    m3ab += _einsum('laKB,iJKl->iJaB', oxOV, tmp) * .5

    u1_ox += _einsum('ijab,jb->ia', m3aa, t1a)
    u1_ox += _einsum('iJaB,JB->ia', m3ab, t1b)
    u1_OX += _einsum('IJAB,JB->IA', m3bb, t1b)
    u1b += _einsum('jIbA,jb->IA', m3ab, t1_ox)

    u2_ooxv += m3aa
    u2_OOXV += m3bb
    u2_oOxV += m3ab
    u2_ooxv += oxov.transpose(0, 2, 1, 3)
    u2_OOXV += OXOV.transpose(0, 2, 1, 3)
    u2_oOxV += oxOV.transpose(0, 2, 1, 3)

    if checkpoint == 15:
        return dict(
            ovOV=oxOV.collect().data,
            tmp=tmp.collect().data,
            m3aa=m3aa.collect().data,
            m3bb=m3bb.collect().data,
            m3ab=m3ab.collect().data,
            u2aa=u2_ooxv.collect().data,
            u2bb=u2_OOXV.collect().data,
            u2ab=u2_oOxV.collect().data,
            u1a = u1a0.data + u1_ox.collect().data + u1a.collect().data,
            u1b = u1b0.data + u1_OX.collect().data + u1b.collect().data,
        )

    # PYSCF L429-L435
    fov1 = _einsum('kcjb,kc->jb', oxov, t1_ox).collect().set(label='fov1aa') + fova
    fov1 += _einsum('jbKC,KC->jb', oxOV, t1b).collect()     # j[b]
    tmp = _einsum('ia,jb->ijab', l1a[:, slc_va], fov1)
    tmp += _einsum('kica,jbck->ijab', l2_ooxv, imds.wovxo, out=tmp)  # [c]a,[c]->[a]
    tmp += _einsum('iKaC,jbCK->ijab', l2_oOxV, imds.woxVO, out=tmp)
    tmp = tmp - tmp.transpose(1, 0, 2, 3)
    u2_ooxv += tmp - tmp.transpose(0, 1, 3, 2)

    fov1 = _einsum('kcjb,kc->jb', OXOV, t1_OX).set(label='fov1bb')
    fov1+= _einsum('kcJB,kc->JB', oxOV, t1_ox)
    fov1 = fov1.collect() + fovb

    tmp = _einsum('ia,jb->ijab', l1b[:, slc_vb], fov1)
    tmp+= _einsum('kica,jbck->ijab', l2_OOXV, imds.wOVXO, out=tmp)
    tmp+= _einsum('kIcA,JBck->IJAB', l2_oOxV, imds.wOVxo, out=tmp)
    tmp = tmp - tmp.transpose(1, 0, 2, 3)
    u2_OOXV += tmp - tmp.transpose(0, 1, 3, 2)

    fov1 = _einsum('kcjb,kc->jb', OXOV, t1_OX).set(label='fov1bb')
    fov1 += _einsum('kcJB,kc->JB', oxOV, t1_ox)
    fov1 = fov1.collect() + fovb

    u2_oOxV += _einsum('ia,JB->iJaB', l1_ox, fov1)
    u2_oOxV += _einsum('iKaC,JBCK->iJaB', l2_oOxV, imds.wOVXO)  # [a]C,[C]->[a]
    u2_oOxV += _einsum('kIaC,jBCk->jIaB', l2_oOxV, imds.woVXo)  # [a]C,[C]->[a]
    u2_oOxV += _einsum('kica,JBck->iJaB', l2_ooxv, imds.wOVxo, out=u2_oOxV)  # [c]a,[c]->[a]
    u2_oOxV += _einsum('iKcA,JbcK->iJbA', l2_oOxV, imds.wOvxO, out=u2_oOxV)  # [c],b[c]->[b]

    fov1 = _einsum('kcjb,kc->jb', oxov, t1_ox).collect().set(label='fov1aa') + fova
    fov1 += _einsum('jbKC,KC->jb', oxOV, t1b).collect()
    u2_oOxV += _einsum('ia,jb->jiba', l1b, fov1[:, slc_va])
    u2_oOxV += _einsum('kIcA,jbck->jIbA', l2_oOxV, imds.wovxo, out=u2_oOxV)  # [c],b[c]->[b]
    u2_oOxV += _einsum('KICA,jbCK->jIbA', l2_OOXV, imds.woxVO)  # [C],b[C]->[b]

    # pyscf L458-L477
    oxoo, OXOO, OXoo, oxOO = map(_get_integral, ("oxoo", "OXOO", "OXoo", "oxOO"))
    oxoo = _fntp(oxoo, 0, 2, 1.0, -1.0)
    OXOO = _fntp(OXOO, 0, 2, 1.0, -1.0)

    tmp = _einsum('ka,jbik->ijab', l1a[:, slc_va], oxoo).set(label='tmp', debug=debug)
    tmp+= _einsum('ijca,cb->ijab', l2_ooxv, v1a[slc_va], out=tmp)  # [c]a,[c]->[a]
    tmp+= _einsum('ca,icjb->ijab', mvv1[:, slc_va], oxov, out=tmp)
    u2_ooxv -= tmp - tmp.transpose(0, 1, 3, 2)

    tmp = _einsum('ka,jbik->ijab', l1b[:, slc_vb], OXOO).set(label='tmp', debug=debug)    # [a],[b]->[a]bb
    tmp += _einsum('ijca,cb->ijab', l2_OOXV, v1b[slc_vb], out=tmp)  # [c]a,[c]->[a]
    tmp += _einsum('ca,icjb->ijab', mVV1[:, slc_vb], OXOV, out=tmp) # c[a],[c]->[a]
    u2_OOXV -= tmp - tmp.transpose(0, 1, 3, 2)

    u2_oOxV -= _einsum('ka,JBik->iJaB', l1_ox, OXoo, out=u2_oOxV)   # [a],[B]->[a]B
    u2_oOxV += _einsum('iJaC,CB->iJaB', l2_oOxV, v1b)   # [a],...->[a]
    u2_oOxV -= _einsum('ca,icJB->iJaB', mvv1[slc_va], oxOV, out=u2_oOxV)    # [c]a,[c]->[a]
    u2_oOxV -= _einsum('KA,ibJK->iJbA', l1b, oxOO)  # ...,[b]->[b]
    u2_oOxV += _einsum('iJcA,cb->iJbA', l2_oOxV, v1a[slc_va], out=u2_oOxV)  # [c],[c]b->[b]
    u2_oOxV -= _einsum('CA,ibJC->iJbA', mVV1, oxOV) # ...,[b]->[b]

    if checkpoint == 20:
        return dict(
            v1a=v1a.data, v1b=v1b.data,
            mvv1=mvv1.data,
            mVV1=mVV1.data,
            u2aa=u2_ooxv.collect().data,
            u2bb=u2_OOXV.collect().data,
            u2ab=u2_oOxV.collect().data,
            u1a = u1a0.data + u1_ox.collect().data + u1a.collect().data,
            u1b = u1b0.data + u1_OX.collect().data + u1b.collect().data,
        )

    # PySCF L478-L506

    u1a0 += fova
    u1b0 += fovb
    u1a0 += _einsum('ib,ba->ia', l1a, v1a)
    u1a0 -= _einsum('ja,ij->ia', l1a, v2a)
    u1b0 += _einsum('ib,ba->ia', l1b, v1b)
    u1b0 -= _einsum('ja,ij->ia', l1b, v2b)

    ovxo, ooxv, oxVO, OVXO, OOXV, OVxo = map(_get_integral, ("ovxo", "ooxv", "oxVO", "OVXO", "OOXV", "OVxo"))
    u1a += _einsum('jb,iabj->ia', l1_ox, ovxo)    # [b],[b]->...
    u1a -= _einsum('jb,ijba->ia', l1_ox, ooxv)    # [b],[b]->...
    u1_ox += _einsum('JB,iaBJ->ia', l1b, oxVO)    # ...,[a]->[a]
    u1b += _einsum('jb,iabj->ia', l1_OX, OVXO)    # [b],[b]->...
    u1b -= _einsum('jb,ijba->ia', l1_OX, OOXV)    # [b],[b]->...
    u1b += _einsum('jb,iabj->ia', l1_ox, OVxo)    # [b],[b]->...

    u1a -= _einsum('kjca,ijck->ia', l2_ooxv, imds.wooxo)
    u1_ox -= _einsum('jKaC,ijCK->ia', l2_oOxV, imds.wooXO)    # [a]C,[C]->[a]
    u1b -= _einsum('kjca,ijck->ia', l2_OOXV, imds.wOOXO)      # [c],[c]->...
    u1b -= _einsum('kJcA,IJck->IA', l2_oOxV, imds.wOOxo)      # [c],[c]->...

    u1a -= _einsum('ikbc,back->ia', l2_ooxv, imds.wvvxo)    # [b]c,b[c]->
    u1_ox -= _einsum('iKbC,baCK->ia', l2_oOxV, imds.wvxVO)    # [b],b[a]->[a]
    u1b -= _einsum('IKBC,BACK->IA', l2_OOXV, imds.wVVXO)    # [B]C,B[C]->...
    u1b -= _einsum('kIcB,BAck->IA', l2_oOxV, imds.wVVxo)    # c,c->...

    u1a += _einsum('jiba,bj->ia', l2_ooxv, imds.w3a[slc_va])    # [b],[b]->...
    u1_ox += _einsum('iJaB,BJ->ia', l2_oOxV, imds.w3b)            # [a],...->[a]
    u1b += _einsum('JIBA,BJ->IA', l2_OOXV, imds.w3b[slc_vb])    # [B],[B]->...
    u1b += _einsum('jIbA,bj->IA', l2_oOxV, imds.w3a[slc_va])    # [b],[b]->...

    tmpa  = t1a + _einsum('kc,kjcb->jb', l1_ox, t2_ooxv).collect()  #[c],[c]->...
    tmpa += _einsum('KC,jKbC->jb', l1b, t2_oOxV).collect()    #j[b]
    tmpa -= _einsum('bd,jd->jb', mvv1, t1a) # rd
    tmpa -= _einsum('lj,lb->jb', moo, t1a)  # rd

    tmpb  =  _einsum('kc,kjcb->jb', l1_OX, t2_OOXV)  # JB
    tmpb += _einsum('kc,kJcB->JB', l1_ox, t2_oOxV) # JB
    tmpb = tmpb.collect()
    tmpb += t1b
    tmpb -= _einsum('bd,jd->jb', mVV1, t1b) # rd
    tmpb -= _einsum('lj,lb->jb', mOO, t1b) # rd

    if checkpoint == 25:
        return dict(
            woovo=imds.wooxo.collect().data,
            wooVO=imds.wooXO.collect().data,
            wOOvo=imds.wOOxo.collect().data,
            wOOVO=imds.wOOXO.collect().data,
            wvvvo=imds.wvvxo.collect().data,
            tmpa=tmpa.data, tmpb=tmpb.data,
            v1a=v1a.data, v1b=v1b.data,
            v2a=v2a.data, v2b=v2b.data,
            w3a=imds.w3a.data,
            w3b=imds.w3b.data,
            u2aa=u2_ooxv.collect().data,
            u2bb=u2_OOXV.collect().data,
            u2ab=u2_oOxV.collect().data,
            u1a = u1a0.data + u1_ox.collect().data + u1a.collect().data,
            u1b = u1b0.data + u1_OX.collect().data + u1b.collect().data,
        )

    u1a += _einsum('jbia,jb->ia', oxov, tmpa[:, slc_va])    # [b],[b]->...
    u1_ox += _einsum('iaJB,JB->ia', oxOV, tmpb)       # [A],...->[A]
    u1b += _einsum('jbia,jb->ia', OXOV, tmpb[:, slc_vb])    # [b],[b]->...
    u1b += _einsum('jbIA,jb->IA', oxOV, tmpa[:, slc_va])    # [b],[b]->...

    u1_ox -= _einsum('iajk,kj->ia', oxoo, moo1)   # i[a]
    u1_ox -= _einsum('iaJK,KJ->ia', oxOO, mOO1)   # i[a]
    u1_OX -= _einsum('iajk,kj->ia', OXOO, mOO1)   # I[A]
    u1_OX -= _einsum('IAjk,kj->IA', OXoo, moo1)   # I[A]

    if checkpoint == 30:
        return dict(
            u2aa=u2_ooxv.collect().data,
            u2bb=u2_OOXV.collect().data,
            u2ab=u2_oOxV.collect().data,
            u1a = u1a0.data + u1_ox.collect().data + u1a.collect().data,
            u1b = u1b0.data + u1_OX.collect().data + u1b.collect().data,
        )

    tmp  = -_einsum('kbja,jb->ka', oxov, t1_ox).collect().set(label='tmp') + fova  # ka
    tmp += _einsum('kaJB,JB->ka', oxOV, t1b).collect()    # k[a]
    u1a0 -= _einsum('ik,ka->ia', moo, tmp)   # rd
    u1a0 -= _einsum('ca,ic->ia', mvv, tmp)   # rd
    tmp  = - _einsum('kbja,jb->ka', OXOV, t1_OX).collect().set(label='tmp') + fovb # ka
    tmp += _einsum('jbKA,jb->KA', oxOV, t1_ox).collect() # [K]A
    u1b0 -= _einsum('ik,ka->ia', mOO, tmp)   # rd
    u1b0 -= _einsum('ca,ic->ia', mVV, tmp)   # rd

    eia = lib.direct_sum('i-j->ij', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-j->ij', mo_eb_o, mo_eb_v)

    if checkpoint == 35:
        return dict(
            tmp=tmp.data,
            mOO=mOO.data,
            mVV=mVV.data,
            u2aa=u2_ooxv.collect().data,
            u2bb=u2_OOXV.collect().data,
            u2ab=u2_oOxV.collect().data,
            u1a = u1a0.data + u1_ox.collect().data + u1a.collect().data,
            u1b = u1b0.data + u1_OX.collect().data + u1b.collect().data,
            eia=eia, eIA=eIA,
        )

    u1a = (u1a0 + u1_ox) + u1a
    u1b = (u1b0 + u1_OX) + u1b
    u1a /= eia
    u1b /= eIA

    u2_ooxv /= lib.direct_sum('ia+jb->ijab', eia[:, slc_va], eia)
    u2_oOxV /= lib.direct_sum('ia+jb->ijab', eia[:, slc_va], eIA)
    u2_OOXV /= lib.direct_sum('ia+jb->ijab', eIA[:, slc_vb], eIA)

    if checkpoint == 36:
        return dict(u1a=u1a.data, u1b=u1b.data,
                    u2aa=u2_ooxv.collect().data,
                    u2bb=u2_OOXV.collect().data,
                    u2ab=u2_oOxV.collect().data)

    if checkpoint is None:
        if return_segarray:
            return (u1a, u1b), (u2_ooxv, u2_oOxV, u2_OOXV)
        else:
            l1new = (u1a.data, u1b.data)
            l2new = (u2_ooxv.data, u2_oOxV.data, u2_OOXV.data)
            return l1new, l2new


@mpi.parallel_call(skip_args=[1], skip_kwargs=['eris'])
def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO,
           fintermediates=None, fupdate=None):
    log = logger.new_logger(mycc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    _sync_(mycc)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None:
        if mycc.l1 is None: l1 = t1
        else: l1 = mycc.l1
    if l2 is None:
        if mycc.l2 is None: l2 = t2
        else: l2 = mycc.l2

    eris = getattr(mycc, '_eris', None)
    if eris is None:
        mycc.ao2mo(mycc.mo_coeff)
        eris = mycc._eris

    if fintermediates is None:
        fintermediates = make_intermediates
    if fupdate is None:
        fupdate = update_lambda

    imds = fintermediates(mycc, t1, t2, eris)

    if isinstance(mycc.diis, diis.DistributedDIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = diis.DistributedDIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput1 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = fupdate(mycc, t1, t2, l1, l2, eris, imds)
        normt = _diff_norm(mycc, l1new, l2new, l1, l2)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        l1, l2 = mycc.run_diis(l1, l2, istep, normt, 0, adiis)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if normt < tol:
            conv = True
            break

    mycc.l1 = l1
    mycc.l2 = l2
    log.timer('CCSD lambda', *cput0)
    return conv, l1, l2