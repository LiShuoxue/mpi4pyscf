"""
MPI-parallelized uccsd_krylov.
"""
from pyscf import lib, __config__
from pyscf.cc import uccsd as pyscf_uccsd
from mpi4pyscf.cc import gccsd_krylov as mpi_gccsd_krylov

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.ccsd import _sync_, _pack_scf, _task_location
from mpi4pyscf.cc import uccsd as mpi_uccsd
from mpi4pyscf.cc import cc_tools as tools


BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)

comm = mpi.comm
rank = mpi.rank
ntasks = mpi.pool.size


def make_precond_vec_finv(mycc, t2, eris, tol=1e-8):
    t2_ooxv, t2_oOxV, _ = t2
    nocca, noccb, _, nvirb = t2_oOxV.shape
    nvira = t2_ooxv.shape[3]
    vlocsa, vlocsb = map(tools.get_vlocs, (nvira, nvirb))
    slc_va = slice(vlocsa[rank][0], vlocsa[rank][1])
    slc_vb = slice(vlocsb[rank][0], vlocsb[rank][1])

    mo_e_oa = eris.mo_energy[0][:nocca]
    mo_e_ob = eris.mo_energy[1][:noccb]
    mo_e_va = eris.mo_energy[0][nocca:] + mycc.level_shift
    mo_e_vb = eris.mo_energy[1][noccb:] + mycc.level_shift
    eia = lib.direct_sum('i-a->ia', mo_e_oa, mo_e_va)
    eIA = lib.direct_sum('i-a->ia', mo_e_ob, mo_e_vb)
    eia[eia > -tol] = -tol
    eIA[eIA > -tol] = -tol

    t1_ov_new = eia
    t1_OV_new = eIA
    t2_ooxv_new = lib.direct_sum('ia+jb->ijab', eia[:, slc_va], eia)
    t2_oOxV_new = lib.direct_sum('ia+jb->ijab', eia[:, slc_va], eIA)
    t2_OOXV_new = lib.direct_sum('ia+jb->ijab', eIA[:, slc_vb], eIA)

    t1_new = (t1_ov_new, t1_OV_new)
    t2_new = (t2_ooxv_new, t2_oOxV_new, t2_OOXV_new)

    res = mycc.amplitudes_to_vector(t1=t1_new, t2=t2_new)
    res = mycc.gather_vector(res)
    return res


def _init_uiccsd_krylov(ccsd_obj):
    from pyscf import gto
    from mpi4pyscf.tools import mpi
    from mpi4pyscf.cc import uccsd_krylov
    if mpi.rank == 0:
        mpi.comm.bcast((ccsd_obj.mol.dumps(), ccsd_obj.pack()))
    else:
        ccsd_obj = uccsd_krylov.UICCSD_KRYLOV.__new__(uccsd_krylov.UICCSD_KRYLOV)
        ccsd_obj.t1 = ccsd_obj.t2 = ccsd_obj.l1 = ccsd_obj.l2 = None
        mol, cc_attr = mpi.comm.bcast(None)
        ccsd_obj.mol = gto.mole.loads(mol)
        ccsd_obj.unpack_(cc_attr)
    if False:  # If also to initialize cc._scf object
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


class UICCSD_KRYLOV(mpi_uccsd.UICCSD):
    """
    MPI version of unrestricted CCSD with Krylov subspace solver.
    """
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 method='krylov', precond='finv', inner_m=10, outer_k=6,
                 frozen_abab=False, nocc_a=None, nvir_a=None):
        pyscf_uccsd.UCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.method = method
        self.precond = precond
        self.inner_m = inner_m
        self.outer_k = outer_k
        self._keys = self._keys.union(["method", "precond", "inner_m", "outer_k"])
        regs = mpi.pool.apply(_init_uiccsd_krylov, (self, ), (None, ))
        self._reg_procs = regs

    def dump_flags(self, verbose=None):
        if rank == 0:
            mpi_uccsd.UICCSD.dump_flags(self, verbose)
            logger.info(self, "method  = %s", self.method)
            logger.info(self, "precond = %s", self.precond)
            logger.info(self, "inner_m = %d", self.inner_m)
            logger.info(self, "outer_k = %d", self.outer_k)
        return self

    def pack(self):
        packed_args = ('verbose', 'max_memory', 'frozen', 'mo_coeff', 'mo_occ',
                       '_nocc', '_nmo', 'diis_file', 'diis_start_cycle',
                       'level_shift', 'direct', 'diis_space',
                       'method', 'precond', 'inner_m', 'outer_k')
        return {arg: getattr(self, arg) for arg in packed_args} 

    def ccsd(self, t1=None, t2=None, eris=None):
        assert self.mo_coeff is not None
        assert self.mo_occ is not None
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        mpi_gccsd_krylov.pre_kernel(self, eris, t1, t2,
            max_cycle=self.max_cycle,
            tol=self.conv_tol, tolnormt=self.conv_tol_normt,
            verbose=self.verbose)

        self.converged, self.eccsd, self.t1, self.t2 = mpi_gccsd_krylov.kernel(self)

        if rank == 0:
            self._finalize()
        return self.e_corr, self.t1, self.t2


    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None, approx_l=False):
        from mpi4pyscf.cc import gccsd_lambda_krylov as glk_mod
        from mpi4pyscf.cc.uccsd_lambda import make_intermediates as f_mi

        glk_mod.pre_kernel(self, eris, t1, t2, l1, l2,
              max_cycle=self.max_cycle,
              tol=self.conv_tol_normt,
              verbose=self.verbose,
              fintermediates=f_mi,
              approx_l=approx_l)

        if approx_l: conv = True
        else:
            conv, self.l1, self.l2 = glk_mod.kernel(self)
        self.converged_lambda = conv
        return self.l1, self.l2

    mop = mpi_gccsd_krylov.mop
    distribute_vector_ = mpi_gccsd_krylov.distribute_vector_
    gather_vector = mpi_gccsd_krylov.gather_vector
    get_res = mpi_gccsd_krylov.get_res
    make_precond_vec_finv = make_precond_vec_finv
