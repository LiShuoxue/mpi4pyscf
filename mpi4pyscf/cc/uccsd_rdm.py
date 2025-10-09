from functools import partial, reduce
import operator
import numpy as np

from pyscf import lib
from pyscf.lib import einsum
from pyscf.cc import uccsd_rdm

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc import uccsd as mpi_uccsd
from mpi4pyscf.cc import cc_tools as tools
from mpi4pyscf.cc.cc_tools import SegArray

# TODO MPI version of Gamma intermediates

rank = mpi.rank


def _gamma1_intermediates(cc, t1, t2, l1, l2):

    debug = (cc.verbose >= logger.DEBUG2)

    (t1a, t1b), (l1a, l1b) = t1, l1
    t2_ooxv, t2_oOxV, t2_OOXV = t2
    l2_ooxv, l2_oOxV, l2_OOXV = l2
    nvira, nvirb = t1a.shape[1], t1b.shape[1]
    nvira_seg = t2_ooxv.shape[2]
    nvirb_seg = t2_OOXV.shape[2]
    vlocs_a, vlocs_b = map(tools.get_vlocs, (nvira, nvirb))
    vloc0a, vloc1a = vlocs_a[rank]
    slc_va = slice(vloc0a, vloc1a)

    _einsum = partial(mpi_uccsd.einsum_sz, nvira_or_vlocsa=vlocs_a, nvirb_or_vlocsb=vlocs_b)

    seg_spins = [0, 0, 1] * 2
    t2_labels = ('t2aa', 't2ab', 't2bb', 'l2aa', 'l2ab', 'l2bb')
    t2_ooxv, t2_oOxV, t2_OOXV, l2_ooxv, l2_oOxV, l2_OOXV = [
        SegArray(data=d, seg_idx=2, seg_spin=ss, label=l, debug=debug, reduced=False)
        for d, ss, l in zip(
            (t2_ooxv, t2_oOxV, t2_OOXV, l2_ooxv, l2_oOxV, l2_OOXV),
            seg_spins,
            t2_labels
        )
    ]

    tmpA  = _einsum("imef,jmef->ij", l2_oOxV, t2_oOxV)   # [e],[e]->...
    tmpA += _einsum("imef,jmef->ij", l2_ooxv, t2_ooxv) * .5

    tmpB  = _einsum("mief,mjef->ij", l2_oOxV, t2_oOxV)  # [e],[e]->...
    tmpB += _einsum("imef,jmef->ij", l2_OOXV, t2_OOXV) * .5

    tmpC  = SegArray(np.zeros((nvira_seg, nvira)), seg_idx=0, seg_spin=0, label='tmpC', debug=debug)
    # print(f"t2_oOxV.shape = {t2_oOxV.shape}, l2_oOxV.shape = {l2_oOxV.shape}, tmpC.shape = {tmpC.shape}")
    tmpC += _einsum("mnae,mnbe->ab", t2_oOxV, l2_oOxV, out=tmpC)  # [a],[b]->[a]b
    tmpC += _einsum("mnae,mnbe->ab", t2_ooxv, l2_ooxv, out=tmpC) * .5

    tmpD  = _einsum("mnea,mneb->ab", t2_oOxV, l2_oOxV).collect()  # [e],[e]->...
    tmpD2 = SegArray(np.zeros((nvirb_seg, nvirb)), seg_idx=0, seg_spin=1, label='tmpD2', debug=debug)
    tmpD2 += _einsum("mnae,mnbe->ab", t2_OOXV, l2_OOXV, out=tmpD2) * .5 # [a],[b]->[a]b
    tmpD += tmpD2.collect()

    tmpA, tmpB, tmpC, tmpD = (x.collect().data for x in (tmpA, tmpB, tmpC, tmpD))

    dooa  = -einsum('ie,je->ij', l1a, t1a)
    dooa -= tmpA
    doob  = -einsum('ie,je->ij', l1b, t1b)
    doob -= tmpB

    dvva  = einsum('ma,mb->ab', t1a, l1a)
    dvva += tmpC
    dvvb  = einsum('ma,mb->ab', t1b, l1b)
    dvvb += tmpD

    xt1a = tmpA
    xt2a = tmpC
    xt2a += einsum('ma,me->ae', t1a, l1a)

    xt1b = tmpB
    xt2b = tmpD
    xt2b += einsum('ma,me->ae', t1b, l1b)

    labels = ('t1a', 't1b', 'l1a', 'l1b')
    t1a, t1b, l1a, l1b = (SegArray(data=d, reduced=True, debug=debug, label=l)
                          for d, l in zip((t1a, t1b, l1a, l1b), labels))

    dvoa  = _einsum('imae,me->ai', t2_ooxv, l1a)    # [a],...->[a]
    dvoa += _einsum('imae,me->ai', t2_oOxV, l1b)
    dvoa = dvoa.collect().data
    dvoa -= einsum('mi,ma->ai', xt1a, t1a.data)
    dvoa -= einsum('ie,ae->ai', t1a.data, xt2a)
    dvoa += t1a.data.T

    l1_ox = l1a[:, slc_va].set(seg_spin=0, label='l1_ox')

    dvob  = _einsum('imae,me->ai', t2_OOXV, l1b).collect().data # [a],...->[a]
    dvob += _einsum('miea,me->ai', t2_oOxV, l1_ox).collect().data  # [e],[e]->...
    dvob -= einsum('mi,ma->ai', xt1b, t1b.data)
    dvob -= einsum('ie,ae->ai', t1b.data, xt2b)
    dvob += t1b.data.T

    dova = l1a.data
    dovb = l1b.data

    return ((dooa, doob), (dova, dovb), (dvoa, dvob), (dvva, dvvb))


def _gamma2_intermediates(cc, t1, t2, l1, l2, compress_vvvv=False):

    debug = (cc.verbose >= logger.DEBUG2)

    t1a, t1b = t1
    l1a, l1b = l1
    t2_ooxv, t2_oOxV, t2_OOXV = t2
    l2_ooxv, l2_oOxV, l2_OOXV = l2
    (nocca, nvira), (noccb, nvirb) = t1a.shape, t1b.shape
    vlocs_a, vlocs_b = map(tools.get_vlocs, (nvira, nvirb))
    nvira_seg, nvirb_seg = t2_ooxv.shape[2], t2_OOXV.shape[2]
    vloc0a, vloc1a = vlocs_a[rank]
    vloc0b, vloc1b = vlocs_b[rank]
    slc_va = slice(vloc0a, vloc1a)
    slc_vb = slice(vloc0b, vloc1b)

    _einsum = partial(mpi_uccsd.einsum_sz, nvira_or_vlocsa=vlocs_a, nvirb_or_vlocsb=vlocs_b)

    def _fn_newarr(astr: str, label=''):
        """"""
        seg_strs = 'wxyzWXYZ'
        label_dict = {'abcd': nvira, 'ijkl': nocca, 'wxyz': nvira_seg,
         'ABCD': nvirb, 'IJKL': noccb, 'WXYZ': nvirb_seg}
        label_dict = reduce(operator.or_, [{s: v for s in ss} for ss, v in label_dict.items()])
        seg_idx = next(i for i, s in enumerate(astr) if s in seg_strs)
        arr_shape = tuple(label_dict[s] for s in astr)
        seg_spin = dict(zip(seg_strs, [0, 0, 0, 0, 1, 1, 1, 1]))[astr[seg_idx]]
        return SegArray(np.zeros(arr_shape), seg_idx=seg_idx, seg_spin=seg_spin, label=label, debug=debug)

    seg_spins = [0, 0, 1] * 2
    labels = ('t2aa', 't2ab', 't2bb', 'l2aa', 'l2ab', 'l2bb')
    t2_ooxv, t2_oOxV, t2_OOXV, l2_ooxv, l2_oOxV, l2_OOXV = [
        SegArray(data=d, seg_idx=2, seg_spin=ss, label=l, debug=debug, reduced=False) for d, ss, l in zip(
            (t2_ooxv, t2_oOxV, t2_OOXV, l2_ooxv, l2_oOxV, l2_OOXV),
            seg_spins,
            labels
        )
    ]

    t1a, t1b, l1a, l1b = (SegArray(data=d, reduced=True, debug=debug, label=l)
                            for d, l in zip((t1a, t1b, l1a, l1b), ('t1a', 't1b', 'l1a', 'l1b')))

    t1_ox = t1a[:, slc_va].set(seg_spin=0, label='t1_ox')
    t1_OX = t1b[:, slc_vb].set(seg_spin=1, label='t1_OX')
    l1_ox = l1a[:, slc_va].set(seg_spin=0, label='l1_ox')
    l1_OX = l1b[:, slc_vb].set(seg_spin=1, label='l1_OX')

    tau_ooxv = t2_ooxv + _einsum('ia,jb->ijab', 2. * t1_ox, t1a)    # [a],...->[a]
    tau_oOxV = t2_oOxV + _einsum("ia,jb->ijab", t1_ox, t1b)
    tau_OOXV = t2_OOXV + _einsum('ia,jb->ijab', 2. * t1_OX, t1b)

    mixjb = _fn_newarr('ixjb', label='miajb')
    mixJB = _fn_newarr('ixJB', label='miaJB')
    mIAjx = _fn_newarr('IAjx', label='mIAjb')
    mIXJB = _fn_newarr('IXJB', label='mIAJB')
    miXjB = _fn_newarr('iXjB', label='miAjB')
    mIxJb = _fn_newarr('IxJb', label='mIaJb')

    mixjb += _einsum('ikac,kjcb->iajb', l2_ooxv, t2_oOxV, out=mixjb)    # [a]c,[c]->[a]
    mixjb += _einsum('ikac,jkbc->iajb', l2_oOxV, t2_oOxV, out=mixjb)    # [a],[b]->[a]b

    mixJB += _einsum('ikac,kjcb->iajb', l2_ooxv, t2_oOxV, out=mixJB)    # [a]c,[c]->[a]
    mixJB += _einsum('ikac,kjcb->iajb', l2_oOxV, t2_OOXV, out=mixJB)    # [a]c,[c]->[a]

    mIAjx += _einsum('kica,jkbc->iajb', l2_OOXV, t2_oOxV, out=mIAjx)    # [c],[b]c->[b]
    mIAjx += _einsum('kica,kjcb->iajb', l2_oOxV, t2_ooxv, out=mIAjx)    # [c],[c]b->[b]

    mIXJB += _einsum('ikac,kjcb->iajb', l2_OOXV, t2_OOXV, out=mIXJB)    # [a]c,[c]->[a]
    mIXJB += _einsum('kica,kjcb->iajb', l2_oOxV, t2_oOxV, out=mIXJB)    # [c]a,[c]->[a]

    miXjB += _einsum('ikca,jkcb->iajb', l2_oOxV, t2_oOxV, out=miXjB) # [c]a,[c]->[a]
    mIxJb += _einsum('kiac,kjbc->iajb', l2_oOxV, t2_oOxV, out=mIxJb) # [a],[b]->[a]b

    goovv = (l2_ooxv.conj() + tau_ooxv) * .25
    goOvV = (l2_oOxV.conj() + tau_oOxV) * .5
    gOOVV = (l2_OOXV.conj() + tau_OOXV) * .25

    tmpa  = _einsum('kc,kica->ia', l1_ox, t2_ooxv).collect()  # [c],[c]->...
    tmpa += _einsum('kc,ikac->ia', l1b, t2_oOxV).collect()    # ...,[a]->[a]
    tmpb  = einsum('kc,kica->ia', l1_OX, t2_OOXV)
    tmpb += einsum('kc,kica->ia', l1_ox, t2_oOxV)
    tmpb = tmpb.collect()

    tmp_vx = tmpa[:, slc_va].set(seg_spin=0, debug=debug)
    tmp_VX = tmpb[:, slc_vb].set(seg_spin=1, debug=debug)
    goovv += _einsum('ia,jb->ijab', tmp_vx, t1a)
    goOvV += _einsum('ia,jb->ijab', tmp_vx, t1b) * .5
    goOvV += _einsum('ia,jb->jiba', tmpb, t1_ox) * .5
    gOOVV += _einsum('ia,jb->ijab', tmp_VX, t1b)

    # pyscf L119-L130
    tmpa = _einsum('kc,kb->cb', l1a, t1a)
    tmpb = _einsum('kc,kb->cb', l1b, t1b)
    tmp_xv = tmpa[slc_va].set(seg_spin=0, debug=debug)
    tmp_XV = tmpb[slc_vb].set(seg_spin=1, debug=debug)
    goovv += _einsum('cb,ijca->ijab', tmp_xv, t2_oOxV, out=goovv) * .5
    goOvV -= _einsum('cb,ijac->ijab', tmpb, t2_oOxV) * .5
    goOvV -= _einsum('cb,jica->jiba', tmp_xv, t2_oOxV, out=goOvV) * .5 # [c]b,[c]->...
    gOOVV += _einsum('cb,ijca->ijab', tmp_XV, t2_OOXV, out=gOOVV) * .5
    tmpa = _einsum('kc,jc->kj', l1a, t1a)
    tmpb = _einsum('kc,jc->kj', l1b, t1b)
    goovv += _einsum('kiab,kj->ijab', tau_ooxv, tmpa) * .5
    goOvV -= _einsum('ikab,kj->ijab', tau_oOxV , tmpb) * .5
    goOvV -= einsum('kiba,kj->jiba', tau_oOxV, tmpa) * .5
    gOOVV += einsum('kiab,kj->ijab', tau_OOXV, tmpb) * .5

    # ldjd->lj contraction by itself not implemented yet 
    raise NotImplementedError("Unfinished.")


@mpi.parallel_call
def make_rdm1(mycc, t1, t2, l1, l2, ao_repr=False, with_frozen=True, with_mf=True):
    r'''
    One-particle spin density matrices dm1a, dm1b in MO basis (the
    occupied-virtual blocks due to the orbital response contribution are not
    included).

    dm1a[p,q] = <q_alpha^\dagger p_alpha>
    dm1b[p,q] = <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    '''
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if l1 is None:
        l1 = mycc.l1
    if l2 is None:
        l2 = mycc.l2
    if l1 is None:
        l1, l2 = mycc.solve_lambda(t1, t2)
    # t1, t2 = mpi_uccsd.gather_amplitudes(mycc, t1, t2)
    # l1, l2 = mpi_uccsd.gather_lambda(mycc, l1, l2)
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    return uccsd_rdm._make_rdm1(mycc, d1, with_frozen=with_frozen, ao_repr=ao_repr, with_mf=with_mf)


@mpi.parallel_call
def make_rdm2(mycc, t1, t2, l1, l2, ao_repr=False, with_frozen=True, with_dm1=True):
    r'''
    One-particle spin density matrices dm1a, dm1b in MO basis (the
    occupied-virtual blocks due to the orbital response contribution are not
    included).

    dm1a[p,q] = <q_alpha^\dagger p_alpha>
    dm1b[p,q] = <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    '''
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if l1 is None:
        l1 = mycc.l1
    if l2 is None:
        l2 = mycc.l2
    if l1 is None:
        l1, l2 = mycc.solve_lambda(t1, t2)
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)

    t1, t2 = mpi_uccsd.gather_amplitudes(mycc, t1, t2)
    l1, l2 = mpi_uccsd.gather_lambda(mycc, l1, l2)
    if rank == 0:
        d2 = uccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)
        return uccsd_rdm._make_rdm2(mycc, d1, d2, with_dm1=with_dm1, with_frozen=with_frozen, ao_repr=ao_repr)
    else:
        return None
