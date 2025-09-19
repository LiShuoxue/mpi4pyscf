"""
parallel_call functions for testing UCCSD class
"""

from mpi4pyscf.cc import uccsd as uccsd_mpi
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.uccsd import UCCSD as UCCSD_MPI
from mpi4pyscf.cc import cc_tools


@mpi.parallel_call
def test_uiccsd_eris(cc: UCCSD_MPI):
    cc.ao2mo(cc.mo_coeff)
    eris = cc._eris

    def _get_index(s: str):
        for k in 'xX':
            if k in s:
                return s.index(k)
        return None

    res = dict()
    for seg_key in eris._eri_keys:
        full_key = seg_key.replace('x', 'v').replace('X', 'V')
        seg_idx = _get_index(seg_key)
        arr_mpi = cc_tools.collect_array(eris.get_integral(seg_key), seg_idx, sum_over=False)
        res[full_key] = arr_mpi

    return res


@mpi.parallel_call
def test_init_amps(cc: UCCSD_MPI):
    cc.ao2mo(cc.mo_coeff)
    E, (t1a, t1b), (t2aa, t2ab, t2bb) = cc.init_amps()
    res = dict(E=E, t1a=t1a, t1b=t1b)
    for k, v in zip(('t2aa', 't2ab', 't2bb'), (t2aa, t2ab, t2bb)):
        res[k] = cc_tools.collect_array(v, seg_idx=2)
    return res


@mpi.parallel_call
def test_update_amps(cc: UCCSD_MPI):
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t1, t2 = cc.init_amps()
    t1, t2 = cc.update_amps(t1, t2, cc._eris)
    (t1a, t1b), (t2aa, t2ab, t2bb) = t1, t2
    res = dict(t1a=t1a, t1b=t1b)
    for k, v in zip(('t2aa', 't2ab', 't2bb'), (t2aa, t2ab, t2bb)):
        res[k] = cc_tools.collect_array(v, seg_idx=2)
    return res


@mpi.parallel_call
def test_make_tau(cc: UCCSD_MPI, t1=None, t2=None):
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    if t1 is None or t2 is None:
        _, t1, t2 = cc.init_amps()
    taus = uccsd_mpi.make_tau(t2, t1, t1)
    keys = ('tauaa', 'taubb', 'tauab')
    res = {k: cc_tools.collect_array(taus[i], seg_idx=2) for i, k in enumerate(keys)}
    return res


@mpi.parallel_call
def test_add_vvvv(cc: UCCSD_MPI):
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t1, t2 = cc.init_amps()
    oovvs = uccsd_mpi._add_vvvv(t1=t1, t2=t2, eris=cc._eris)
    keys = ('oovv_aa', 'oovv_ab', 'oovv_bb')
    res = {k: cc_tools.collect_array(oovvs[i], seg_idx=2) for i, k in enumerate(keys)}
    return res
