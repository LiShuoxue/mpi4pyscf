"""
backend: parallel_call functions for testing UCCSD class
"""
import numpy as np
from mpi4pyscf.cc import uccsd as uccsd_mpi
from mpi4pyscf.cc import uccsd_lambda as uccsd_lambda_mpi
from mpi4pyscf.tools import mpi
from mpi4pyscf.cc.uccsd import UCCSD as UCCSD_MPI
from mpi4pyscf.cc import cc_tools
from numpy.typing import ArrayLike
from mpi4pyscf.cc.cc_tools import SegArray
import cProfile

tools = cc_tools

rank = mpi.rank
comm = mpi.comm


def update_amps_checkpoint_serial(cc, t1, t2, eris, checkpoint: int | None = None):
    from pyscf import lib
    from pyscf.lib import logger
    from pyscf.cc import ccsd
    from pyscf.cc.uccsd import make_tau, make_tau_aa, make_tau_ab

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(cc.stdout, cc.verbose)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + cc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + cc.level_shift
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]

    u1a = np.zeros_like(t1a)
    u1b = np.zeros_like(t1b)
    #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:u2aa += lib.einsum('ijef,aebf->ijab', tauaa, eris_vvvv) * .5
    #:u2bb += lib.einsum('ijef,aebf->ijab', taubb, eris_VVVV) * .5
    #:u2ab += lib.einsum('iJeF,aeBF->iJaB', tauab, eris_vvVV)
    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    u2aa, u2ab, u2bb = cc._add_vvvv(None, (tauaa,tauab,taubb), eris)
    u2aa *= .5
    u2bb *= .5

    Fooa =  .5 * lib.einsum('me,ie->mi', fova, t1a)
    Foob =  .5 * lib.einsum('me,ie->mi', fovb, t1b)
    Fvva = -.5 * lib.einsum('me,ma->ae', fova, t1a)
    Fvvb = -.5 * lib.einsum('me,ma->ae', fovb, t1b)
    Fooa += eris.focka[:nocca,:nocca] - np.diag(mo_ea_o)
    Foob += eris.fockb[:noccb,:noccb] - np.diag(mo_eb_o)
    Fvva += eris.focka[nocca:,nocca:] - np.diag(mo_ea_v)
    Fvvb += eris.fockb[noccb:,noccb:] - np.diag(mo_eb_v)
    dtype = u2aa.dtype
    wovvo = np.zeros((nocca,nvira,nvira,nocca), dtype=dtype)
    wOVVO = np.zeros((noccb,nvirb,nvirb,noccb), dtype=dtype)
    woVvO = np.zeros((nocca,nvirb,nvira,noccb), dtype=dtype)
    woVVo = np.zeros((nocca,nvirb,nvirb,nocca), dtype=dtype)
    wOvVo = np.zeros((noccb,nvira,nvirb,nocca), dtype=dtype)
    wOvvO = np.zeros((noccb,nvira,nvira,noccb), dtype=dtype)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now - u2aa.size*8e-6)
    if nvira > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            Fvva += np.einsum('mf,mfae->ae', t1a[p0:p1], ovvv)
            wovvo[p0:p1] += lib.einsum('jf,mebf->mbej', t1a, ovvv)
            u1a += 0.5*lib.einsum('mief,meaf->ia', t2aa[p0:p1], ovvv)
            u2aa[:,p0:p1] += lib.einsum('ie,mbea->imab', t1a, ovvv.conj())
            tmp1aa = lib.einsum('ijef,mebf->ijmb', tauaa, ovvv)
            u2aa -= lib.einsum('ijmb,ma->ijab', tmp1aa, t1a[p0:p1]*.5)
            ovvv = tmp1aa = None

    if nvirb > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            Fvvb += np.einsum('mf,mfae->ae', t1b[p0:p1], OVVV)
            wOVVO[p0:p1] = lib.einsum('jf,mebf->mbej', t1b, OVVV)
            u1b += 0.5*lib.einsum('MIEF,MEAF->IA', t2bb[p0:p1], OVVV)
            u2bb[:,p0:p1] += lib.einsum('ie,mbea->imab', t1b, OVVV.conj())
            tmp1bb = lib.einsum('ijef,mebf->ijmb', taubb, OVVV)
            u2bb -= lib.einsum('ijmb,ma->ijab', tmp1bb, t1b[p0:p1]*.5)
            OVVV = tmp1bb = None

    if nvirb > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            Fvvb += np.einsum('mf,mfAE->AE', t1a[p0:p1], ovVV)
            woVvO[p0:p1] = lib.einsum('JF,meBF->mBeJ', t1b, ovVV)
            woVVo[p0:p1] = lib.einsum('jf,mfBE->mBEj',-t1a, ovVV)
            u1b += lib.einsum('mIeF,meAF->IA', t2ab[p0:p1], ovVV)
            u2ab[p0:p1] += lib.einsum('IE,maEB->mIaB', t1b, ovVV.conj())
            tmp1ab = lib.einsum('iJeF,meBF->iJmB', tauab, ovVV)
            u2ab -= lib.einsum('iJmB,ma->iJaB', tmp1ab, t1a[p0:p1])
            ovVV = tmp1ab = None

    if nvira > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            Fvva += np.einsum('MF,MFae->ae', t1b[p0:p1], OVvv)
            wOvVo[p0:p1] = lib.einsum('jf,MEbf->MbEj', t1a, OVvv)
            wOvvO[p0:p1] = lib.einsum('JF,MFbe->MbeJ',-t1b, OVvv)
            u1a += lib.einsum('iMfE,MEaf->ia', t2ab[:,p0:p1], OVvv)
            u2ab[:,p0:p1] += lib.einsum('ie,MBea->iMaB', t1a, OVvv.conj())
            tmp1abba = lib.einsum('iJeF,MFbe->iJbM', tauab, OVvv)
            u2ab -= lib.einsum('iJbM,MA->iJbA', tmp1abba, t1b[p0:p1])
            OVvv = tmp1abba = None

    if checkpoint == 10:
        return dict(u1a=u1a, u1b=u1b)

    eris_ovov = np.asarray(eris.ovov)
    eris_ovoo = np.asarray(eris.ovoo)
    Woooo = lib.einsum('je,nemi->mnij', t1a, eris_ovoo)
    Woooo = Woooo - Woooo.transpose(0,1,3,2)
    Woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
    Woooo += lib.einsum('ijef,menf->mnij', tauaa, eris_ovov) * .5
    u2aa += lib.einsum('mnab,mnij->ijab', tauaa, Woooo*.5)
    Woooo = tauaa = None
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    Fooa += np.einsum('ne,nemi->mi', t1a, ovoo)
    u1a += 0.5*lib.einsum('mnae,meni->ia', t2aa, ovoo)
    wovvo += lib.einsum('nb,nemj->mbej', t1a, ovoo)
    ovoo = eris_ovoo = None

    tilaa = make_tau_aa(t2[0], t1a, t1a, fac=0.5)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    Fvva -= .5 * lib.einsum('mnaf,menf->ae', tilaa, ovov)
    Fooa += .5 * lib.einsum('inef,menf->mi', tilaa, ovov)
    Fova = np.einsum('nf,menf->me',t1a, ovov)
    u2aa += ovov.conj().transpose(0,2,1,3) * .5
    wovvo -= 0.5*lib.einsum('jnfb,menf->mbej', t2aa, ovov)
    woVvO += 0.5*lib.einsum('nJfB,menf->mBeJ', t2ab, ovov)
    tmpaa = lib.einsum('jf,menf->mnej', t1a, ovov)
    wovvo -= lib.einsum('nb,mnej->mbej', t1a, tmpaa)
    eris_ovov = ovov = tmpaa = tilaa = None

    eris_OVOV = np.asarray(eris.OVOV)
    eris_OVOO = np.asarray(eris.OVOO)
    WOOOO = lib.einsum('je,nemi->mnij', t1b, eris_OVOO)
    WOOOO = WOOOO - WOOOO.transpose(0,1,3,2)
    WOOOO += np.asarray(eris.OOOO).transpose(0,2,1,3)
    WOOOO += lib.einsum('ijef,menf->mnij', taubb, eris_OVOV) * .5
    u2bb += lib.einsum('mnab,mnij->ijab', taubb, WOOOO*.5)
    WOOOO = taubb = None
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    Foob += np.einsum('ne,nemi->mi', t1b, OVOO)
    u1b += 0.5*lib.einsum('mnae,meni->ia', t2bb, OVOO)
    wOVVO += lib.einsum('nb,nemj->mbej', t1b, OVOO)
    OVOO = eris_OVOO = None

    tilbb = make_tau_aa(t2[2], t1b, t1b, fac=0.5)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    Fvvb -= .5 * lib.einsum('MNAF,MENF->AE', tilbb, OVOV)
    Foob += .5 * lib.einsum('inef,menf->mi', tilbb, OVOV)
    Fovb = np.einsum('nf,menf->me',t1b, OVOV)
    u2bb += OVOV.conj().transpose(0,2,1,3) * .5
    wOVVO -= 0.5*lib.einsum('jnfb,menf->mbej', t2bb, OVOV)
    wOvVo += 0.5*lib.einsum('jNbF,MENF->MbEj', t2ab, OVOV)
    tmpbb = lib.einsum('jf,menf->mnej', t1b, OVOV)
    wOVVO -= lib.einsum('nb,mnej->mbej', t1b, tmpbb)
    eris_OVOV = OVOV = tmpbb = tilbb = None

    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    Fooa += np.einsum('NE,NEmi->mi', t1b, eris_OVoo)
    u1a -= lib.einsum('nMaE,MEni->ia', t2ab, eris_OVoo)
    wOvVo -= lib.einsum('nb,MEnj->MbEj', t1a, eris_OVoo)
    woVVo += lib.einsum('NB,NEmj->mBEj', t1b, eris_OVoo)
    Foob += np.einsum('ne,neMI->MI', t1a, eris_ovOO)
    u1b -= lib.einsum('mNeA,meNI->IA', t2ab, eris_ovOO)
    woVvO -= lib.einsum('NB,meNJ->mBeJ', t1b, eris_ovOO)
    wOvvO += lib.einsum('nb,neMJ->MbeJ', t1a, eris_ovOO)
    WoOoO = lib.einsum('JE,NEmi->mNiJ', t1b, eris_OVoo)
    WoOoO+= lib.einsum('je,neMI->nMjI', t1a, eris_ovOO)
    WoOoO += np.asarray(eris.ooOO).transpose(0,2,1,3)
    eris_OVoo = eris_ovOO = None

    eris_ovOV = np.asarray(eris.ovOV)
    WoOoO += lib.einsum('iJeF,meNF->mNiJ', tauab, eris_ovOV)
    u2ab += lib.einsum('mNaB,mNiJ->iJaB', tauab, WoOoO)
    WoOoO = None

    if checkpoint == 20:
        return dict(u1a=u1a, u1b=u1b)

    tilab = make_tau_ab(t2[1], t1 , t1 , fac=0.5)
    Fvva -= lib.einsum('mNaF,meNF->ae', tilab, eris_ovOV)
    Fvvb -= lib.einsum('nMfA,nfME->AE', tilab, eris_ovOV)
    Fooa += lib.einsum('iNeF,meNF->mi', tilab, eris_ovOV)
    Foob += lib.einsum('nIfE,nfME->MI', tilab, eris_ovOV)
    Fova += np.einsum('NF,meNF->me',t1b, eris_ovOV)
    Fovb += np.einsum('nf,nfME->ME',t1a, eris_ovOV)
    u2ab += eris_ovOV.conj().transpose(0,2,1,3)
    wovvo += 0.5*lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
    wOVVO += 0.5*lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
    wOvVo -= 0.5*lib.einsum('jnfb,nfME->MbEj', t2aa, eris_ovOV)
    woVvO -= 0.5*lib.einsum('JNFB,meNF->mBeJ', t2bb, eris_ovOV)
    woVVo += 0.5*lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
    wOvvO += 0.5*lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
    tmpabab = lib.einsum('JF,meNF->mNeJ', t1b, eris_ovOV)
    tmpbaba = lib.einsum('jf,nfME->MnEj', t1a, eris_ovOV)
    woVvO -= lib.einsum('NB,mNeJ->mBeJ', t1b, tmpabab)
    wOvVo -= lib.einsum('nb,MnEj->MbEj', t1a, tmpbaba)
    woVVo += lib.einsum('NB,NmEj->mBEj', t1b, tmpbaba)
    wOvvO += lib.einsum('nb,nMeJ->MbeJ', t1a, tmpabab)
    tmpabab = tmpbaba = tilab = None

    if checkpoint == 24:
        return dict(u1a=u1a, u1b=u1b)

    Fova += fova
    Fovb += fovb
    u1a += fova.conj()
    u1a += np.einsum('ie,ae->ia', t1a, Fvva)
    u1a -= np.einsum('ma,mi->ia', t1a, Fooa)
    u1a -= np.einsum('imea,me->ia', t2aa, Fova)
    u1a += np.einsum('iMaE,ME->ia', t2ab, Fovb)
    u1b += fovb.conj()
    u1b += np.einsum('ie,ae->ia',t1b,Fvvb)
    u1b -= np.einsum('ma,mi->ia',t1b,Foob)
    u1b -= np.einsum('imea,me->ia', t2bb, Fovb)
    u1b += np.einsum('mIeA,me->IA', t2ab, Fova)

    if checkpoint == 25:
        return dict(u1a=u1a, u1b=u1b,
                    Fova=Fova, Fovb=Fovb,
                    Fvva=Fvva, Fvvb=Fvvb,
                    Fooa=Fooa, Foob=Foob,
                    u2aa=u2aa, u2ab=u2ab, u2bb=u2bb)

    eris_oovv = np.asarray(eris.oovv)
    eris_ovvo = np.asarray(eris.ovvo)
    wovvo -= eris_oovv.transpose(0,2,3,1)
    wovvo += eris_ovvo.transpose(0,2,1,3)
    oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
    u1a-= np.einsum('nf,niaf->ia', t1a,      oovv)
    tmp1aa = lib.einsum('ie,mjbe->mbij', t1a,      oovv)
    u2aa += 2*lib.einsum('ma,mbij->ijab', t1a, tmp1aa)
    eris_ovvo = eris_oovv = oovv = tmp1aa = None

    eris_OOVV = np.asarray(eris.OOVV)
    eris_OVVO = np.asarray(eris.OVVO)
    wOVVO -= eris_OOVV.transpose(0,2,3,1)
    wOVVO += eris_OVVO.transpose(0,2,1,3)
    OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
    u1b-= np.einsum('nf,niaf->ia', t1b,      OOVV)
    tmp1bb = lib.einsum('ie,mjbe->mbij', t1b,      OOVV)
    u2bb += 2*lib.einsum('ma,mbij->ijab', t1b, tmp1bb)
    eris_OVVO = eris_OOVV = OOVV = None

    eris_ooVV = np.asarray(eris.ooVV)
    eris_ovVO = np.asarray(eris.ovVO)
    woVVo -= eris_ooVV.transpose(0,2,3,1)
    woVvO += eris_ovVO.transpose(0,2,1,3)
    u1b+= np.einsum('nf,nfAI->IA', t1a, eris_ovVO)
    tmp1ab = lib.einsum('ie,meBJ->mBiJ', t1a, eris_ovVO)
    tmp1ab+= lib.einsum('IE,mjBE->mBjI', t1b, eris_ooVV)
    u2ab -= lib.einsum('ma,mBiJ->iJaB', t1a, tmp1ab)
    eris_ooVV = eris_ovVO = tmp1ab = None

    eris_OOvv = np.asarray(eris.OOvv)
    eris_OVvo = np.asarray(eris.OVvo)
    wOvvO -= eris_OOvv.transpose(0,2,3,1)
    wOvVo += eris_OVvo.transpose(0,2,1,3)
    u1a+= np.einsum('NF,NFai->ia', t1b, eris_OVvo)
    tmp1ba = lib.einsum('IE,MEbj->MbIj', t1b, eris_OVvo)
    tmp1ba+= lib.einsum('ie,MJbe->MbJi', t1a, eris_OOvv)
    u2ab -= lib.einsum('MA,MbIj->jIbA', t1b, tmp1ba)
    eris_OOvv = eris_OVvo = tmp1ba = None

    if checkpoint == 30:
        return dict(u1a=u1a, u1b=u1b,
                    u2aa=u2aa, u2ab=u2ab, u2bb=u2bb)

    u2aa += 2*lib.einsum('imae,mbej->ijab', t2aa, wovvo)
    u2aa += 2*lib.einsum('iMaE,MbEj->ijab', t2ab, wOvVo)
    u2bb += 2*lib.einsum('imae,mbej->ijab', t2bb, wOVVO)
    u2bb += 2*lib.einsum('mIeA,mBeJ->IJAB', t2ab, woVvO)
    u2ab += lib.einsum('imae,mBeJ->iJaB', t2aa, woVvO)
    u2ab += lib.einsum('iMaE,MBEJ->iJaB', t2ab, wOVVO)
    u2ab += lib.einsum('iMeA,MbeJ->iJbA', t2ab, wOvvO)
    u2ab += lib.einsum('IMAE,MbEj->jIbA', t2bb, wOvVo)
    u2ab += lib.einsum('mIeA,mbej->jIbA', t2ab, wovvo)
    u2ab += lib.einsum('mIaE,mBEj->jIaB', t2ab, woVVo)

    wovvo = wOVVO = woVvO = wOvVo = woVVo = wOvvO = None

    if checkpoint == 35:
        return dict(u1a=u1a, u1b=u1b,
                    u2aa=u2aa, u2ab=u2ab, u2bb=u2bb)

    Ftmpa = Fvva - .5*lib.einsum('mb,me->be', t1a, Fova)
    Ftmpb = Fvvb - .5*lib.einsum('mb,me->be', t1b, Fovb)
    u2aa += lib.einsum('ijae,be->ijab', t2aa, Ftmpa)
    u2bb += lib.einsum('ijae,be->ijab', t2bb, Ftmpb)
    u2ab += lib.einsum('iJaE,BE->iJaB', t2ab, Ftmpb)
    u2ab += lib.einsum('iJeA,be->iJbA', t2ab, Ftmpa)
    Ftmpa = Fooa + 0.5*lib.einsum('je,me->mj', t1a, Fova)
    Ftmpb = Foob + 0.5*lib.einsum('je,me->mj', t1b, Fovb)
    u2aa -= lib.einsum('imab,mj->ijab', t2aa, Ftmpa)
    u2bb -= lib.einsum('imab,mj->ijab', t2bb, Ftmpb)
    u2ab -= lib.einsum('iMaB,MJ->iJaB', t2ab, Ftmpb)
    u2ab -= lib.einsum('mIaB,mj->jIaB', t2ab, Ftmpa)

    eris_ovoo = np.asarray(eris.ovoo).conj()
    eris_OVOO = np.asarray(eris.OVOO).conj()
    eris_OVoo = np.asarray(eris.OVoo).conj()
    eris_ovOO = np.asarray(eris.ovOO).conj()
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    u2aa -= lib.einsum('ma,jbim->ijab', t1a, ovoo)
    u2bb -= lib.einsum('ma,jbim->ijab', t1b, OVOO)
    u2ab -= lib.einsum('ma,JBim->iJaB', t1a, eris_OVoo)
    u2ab -= lib.einsum('MA,ibJM->iJbA', t1b, eris_ovOO)

    if checkpoint == 40:
        return dict(u1a=u1a, u1b=u1b,
                    u2aa=u2aa, u2ab=u2ab, u2bb=u2bb,
                    ovoo=ovoo, OVOO=OVOO,
                    ovOO=eris_ovOO, OVoo=eris_OVoo,
                    Ftmpa=Ftmpa, Ftmpb=Ftmpb,
                    Fooa=Fooa, Foob=Foob,
                    Fvva=Fvva, Fvvb=Fvvb,
                    Fova=Fova, Fovb=Fovb,
                    t1a=t1a, t1b=t1b)

    eris_ovoo = eris_OVoo = eris_OVOO = eris_ovOO = None

    u2aa *= .5
    u2bb *= .5
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    u2bb = u2bb - u2bb.transpose(1,0,2,3)

    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    u1a /= eia_a
    u1b /= eia_b

    u2aa /= lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    u2ab /= lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    u2bb /= lib.direct_sum('ia+jb->ijab', eia_b, eia_b)

    time0 = log.timer_debug1('update t1 t2', *time0)
    t1new = u1a, u1b
    t2new = u2aa, u2ab, u2bb
    return t1new, t2new


def make_intermediates_checkpoint_serial(mycc, t1, t2, eris, checkpoint=10):
    import numpy
    from pyscf.lib import einsum
    from pyscf.cc import uccsd
    from pyscf import lib
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape

    fooa = eris.focka[:nocca,:nocca]
    fova = eris.focka[:nocca,nocca:]
    fvoa = eris.focka[nocca:,:nocca]
    fvva = eris.focka[nocca:,nocca:]
    foob = eris.fockb[:noccb,:noccb]
    fovb = eris.fockb[:noccb,noccb:]
    fvob = eris.fockb[noccb:,:noccb]
    fvvb = eris.fockb[noccb:,noccb:]

    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)

    ovov = numpy.asarray(eris.ovov)
    ovov = ovov - ovov.transpose(0,3,2,1)
    OVOV = numpy.asarray(eris.OVOV)
    OVOV = OVOV - OVOV.transpose(0,3,2,1)
    ovOV = numpy.asarray(eris.ovOV)

    v1a  = fvva - einsum('ja,jb->ba', fova, t1a)
    v1b  = fvvb - einsum('ja,jb->ba', fovb, t1b)
    v1a += einsum('jcka,jkbc->ba', ovov, tauaa) * .5
    v1a -= einsum('jaKC,jKbC->ba', ovOV, tauab) * .5
    v1a -= einsum('kaJC,kJbC->ba', ovOV, tauab) * .5
    v1b += einsum('jcka,jkbc->ba', OVOV, taubb) * .5
    v1b -= einsum('kcJA,kJcB->BA', ovOV, tauab) * .5
    v1b -= einsum('jcKA,jKcB->BA', ovOV, tauab) * .5

    v2a  = fooa + einsum('ib,jb->ij', fova, t1a)
    v2b  = foob + einsum('ib,jb->ij', fovb, t1b)
    v2a += einsum('ibkc,jkbc->ij', ovov, tauaa) * .5
    v2a += einsum('ibKC,jKbC->ij', ovOV, tauab)
    v2b += einsum('ibkc,jkbc->ij', OVOV, taubb) * .5
    v2b += einsum('kcIB,kJcB->IJ', ovOV, tauab)

    ovoo = numpy.asarray(eris.ovoo)
    ovoo = ovoo - ovoo.transpose(2,1,0,3)
    OVOO = numpy.asarray(eris.OVOO)
    OVOO = OVOO - OVOO.transpose(2,1,0,3)
    OVoo = numpy.asarray(eris.OVoo)
    ovOO = numpy.asarray(eris.ovOO)
    v2a -= numpy.einsum('ibkj,kb->ij', ovoo, t1a)
    v2a += numpy.einsum('KBij,KB->ij', OVoo, t1b)
    v2b -= numpy.einsum('ibkj,kb->ij', OVOO, t1b)
    v2b += numpy.einsum('kbIJ,kb->IJ', ovOO, t1a)

    v5a  = fvoa + numpy.einsum('kc,jkbc->bj', fova, t2aa)
    v5a += numpy.einsum('KC,jKbC->bj', fovb, t2ab)
    v5b  = fvob + numpy.einsum('kc,jkbc->bj', fovb, t2bb)
    v5b += numpy.einsum('kc,kJcB->BJ', fova, t2ab)
    tmp  = fova - numpy.einsum('kdlc,ld->kc', ovov, t1a)
    tmp += numpy.einsum('kcLD,LD->kc', ovOV, t1b)
    v5a += einsum('kc,kb,jc->bj', tmp, t1a, t1a)
    tmp  = fovb - numpy.einsum('kdlc,ld->kc', OVOV, t1b)
    tmp += numpy.einsum('ldKC,ld->KC', ovOV, t1a)
    v5b += einsum('kc,kb,jc->bj', tmp, t1b, t1b)
    v5a -= einsum('lckj,klbc->bj', ovoo, t2aa) * .5
    v5a -= einsum('LCkj,kLbC->bj', OVoo, t2ab)
    v5b -= einsum('LCKJ,KLBC->BJ', OVOO, t2bb) * .5
    v5b -= einsum('lcKJ,lKcB->BJ', ovOO, t2ab)

    oooo = numpy.asarray(eris.oooo)
    OOOO = numpy.asarray(eris.OOOO)
    ooOO = numpy.asarray(eris.ooOO)
    woooo  = einsum('icjl,kc->ikjl', ovoo, t1a)
    wOOOO  = einsum('icjl,kc->ikjl', OVOO, t1b)
    wooOO  = einsum('icJL,kc->ikJL', ovOO, t1a)
    wooOO += einsum('JCil,KC->ilJK', OVoo, t1b)

    woooo += (oooo - oooo.transpose(0,3,2,1)) * .5
    wOOOO += (OOOO - OOOO.transpose(0,3,2,1)) * .5
    wooOO += ooOO.copy()

    if checkpoint == 4:
        return dict(woooo=woooo, wOOOO=wOOOO, wooOO=wooOO,
                    oooo=oooo, OOOO=OOOO)

    woooo += einsum('icjd,klcd->ikjl', ovov, tauaa) * .25
    wOOOO += einsum('icjd,klcd->ikjl', OVOV, taubb) * .25
    wooOO += einsum('icJD,kLcD->ikJL', ovOV, tauab)

    if checkpoint == 5:
        return dict(ovoo=ovoo, OVOO=OVOO,
                    ovOO=ovOO, OVoo=OVoo,
                    ovov=ovov, OVOV=OVOV, ovOV=ovOV,
                    tauaa=tauaa, tauab=tauab, taubb=taubb,
                    woooo=woooo, wOOOO=wOOOO, wooOO=wooOO,)

    v4ovvo  = einsum('jbld,klcd->jbck', ovov, t2aa)
    v4ovvo += einsum('jbLD,kLcD->jbck', ovOV, t2ab)
    v4ovvo += numpy.asarray(eris.ovvo)
    v4ovvo -= numpy.asarray(eris.oovv).transpose(0,3,2,1)
    v4OVVO  = einsum('jbld,klcd->jbck', OVOV, t2bb)
    v4OVVO += einsum('ldJB,lKdC->JBCK', ovOV, t2ab)
    v4OVVO += numpy.asarray(eris.OVVO)
    v4OVVO -= numpy.asarray(eris.OOVV).transpose(0,3,2,1)
    v4OVvo  = einsum('ldJB,klcd->JBck', ovOV, t2aa)
    v4OVvo += einsum('JBLD,kLcD->JBck', OVOV, t2ab)
    v4OVvo += numpy.asarray(eris.OVvo)
    v4ovVO  = einsum('jbLD,KLCD->jbCK', ovOV, t2bb)
    v4ovVO += einsum('jbld,lKdC->jbCK', ovov, t2ab)
    v4ovVO += numpy.asarray(eris.ovVO)
    v4oVVo  = einsum('jdLB,kLdC->jBCk', ovOV, t2ab)
    v4oVVo -= numpy.asarray(eris.ooVV).transpose(0,3,2,1)
    v4OvvO  = einsum('lbJD,lKcD->JbcK', ovOV, t2ab)
    v4OvvO -= numpy.asarray(eris.OOvv).transpose(0,3,2,1)

    woovo  = einsum('ibck,jb->ijck', v4ovvo, t1a)
    wOOVO  = einsum('ibck,jb->ijck', v4OVVO, t1b)
    wOOvo  = einsum('IBck,JB->IJck', v4OVvo, t1b)
    wOOvo -= einsum('IbcK,jb->IKcj', v4OvvO, t1a)
    wooVO  = einsum('ibCK,jb->ijCK', v4ovVO, t1a)
    wooVO -= einsum('iBCk,JB->ikCJ', v4oVVo, t1b)
    woovo += ovoo.conj().transpose(3,2,1,0) * .5
    wOOVO += OVOO.conj().transpose(3,2,1,0) * .5
    wooVO += OVoo.conj().transpose(3,2,1,0)
    wOOvo += ovOO.conj().transpose(3,2,1,0)
    woovo -= einsum('iclk,jlbc->ikbj', ovoo, t2aa)
    woovo += einsum('LCik,jLbC->ikbj', OVoo, t2ab)
    wOOVO -= einsum('iclk,jlbc->ikbj', OVOO, t2bb)
    wOOVO += einsum('lcIK,lJcB->IKBJ', ovOO, t2ab)
    wooVO -= einsum('iclk,lJcB->ikBJ', ovoo, t2ab)
    wooVO += einsum('LCik,JLBC->ikBJ', OVoo, t2bb)
    wooVO -= einsum('icLK,jLcB->ijBK', ovOO, t2ab)
    wOOvo -= einsum('ICLK,jLbC->IKbj', OVOO, t2ab)
    wOOvo += einsum('lcIK,jlbc->IKbj', ovOO, t2aa)
    wOOvo -= einsum('IClk,lJbC->IJbk', OVoo, t2ab)

    wvvvo  = einsum('jack,jb->back', v4ovvo, t1a)
    wVVVO  = einsum('jack,jb->back', v4OVVO, t1b)
    wVVvo  = einsum('JAck,JB->BAck', v4OVvo, t1b)
    wVVvo -= einsum('jACk,jb->CAbk', v4oVVo, t1a)
    wvvVO  = einsum('jaCK,jb->baCK', v4ovVO, t1a)
    wvvVO -= einsum('JacK,JB->caBK', v4OvvO, t1b)
    wvvvo += einsum('lajk,jlbc->back', .25*ovoo, tauaa)
    wVVVO += einsum('lajk,jlbc->back', .25*OVOO, taubb)
    wVVvo -= einsum('LAjk,jLcB->BAck', OVoo, tauab)
    wvvVO -= einsum('laJK,lJbC->baCK', ovOO, tauab)

    w3a  = numpy.einsum('jbck,jb->ck', v4ovvo, t1a)
    w3a += numpy.einsum('JBck,JB->ck', v4OVvo, t1b)
    w3b  = numpy.einsum('jbck,jb->ck', v4OVVO, t1b)
    w3b += numpy.einsum('jbCK,jb->CK', v4ovVO, t1a)

    if checkpoint == 10:
        return dict(
                    v1a=v1a, v1b=v1b,
                    v4OVvo=v4OVvo,
                    v4ovvo=v4ovvo,
                    w3a=w3a, w3b=w3b,
                    wVVvo=wVVvo,
                    OVvo=eris.OVvo,
                    )

    wovvo  = v4ovvo
    wOVVO  = v4OVVO
    wovVO  = v4ovVO
    wOVvo  = v4OVvo
    woVVo  = v4oVVo
    wOvvO  = v4OvvO
    wovvo += lib.einsum('jbld,kd,lc->jbck', ovov, t1a, -t1a)
    wOVVO += lib.einsum('jbld,kd,lc->jbck', OVOV, t1b, -t1b)
    wovVO += lib.einsum('jbLD,KD,LC->jbCK', ovOV, t1b, -t1b)
    wOVvo += lib.einsum('ldJB,kd,lc->JBck', ovOV, t1a, -t1a)
    woVVo += lib.einsum('jdLB,kd,LC->jBCk', ovOV, t1a,  t1b)
    wOvvO += lib.einsum('lbJD,KD,lc->JbcK', ovOV, t1b,  t1a)
    wovvo -= einsum('jblk,lc->jbck', ovoo, t1a)
    wOVVO -= einsum('jblk,lc->jbck', OVOO, t1b)
    wovVO -= einsum('jbLK,LC->jbCK', ovOO, t1b)
    wOVvo -= einsum('JBlk,lc->JBck', OVoo, t1a)
    woVVo += einsum('LBjk,LC->jBCk', OVoo, t1b)
    wOvvO += einsum('lbJK,lc->JbcK', ovOO, t1a)

    if checkpoint == 15:
        return dict(v1a=v1a, v1b=v1b)

    if nvira > 0 and nocca > 0:
        ovvv = numpy.asarray(eris.get_ovvv())
        ovvv = ovvv - ovvv.transpose(0,3,2,1)
        v1a -= numpy.einsum('jabc,jc->ba', ovvv, t1a)
        v5a += einsum('kdbc,jkcd->bj', ovvv, t2aa) * .5
        woovo += einsum('idcb,kjbd->ijck', ovvv, tauaa) * .25
        wovvo += einsum('jbcd,kd->jbck', ovvv, t1a)
        wvvvo -= ovvv.conj().transpose(3,2,1,0) * .5
        wvvvo += einsum('jacd,kjbd->cabk', ovvv, t2aa)
        wvvVO += einsum('jacd,jKdB->caBK', ovvv, t2ab)
        ovvv = tmp = None

    if nvirb > 0 and noccb > 0:
        OVVV = numpy.asarray(eris.get_OVVV())
        OVVV = OVVV - OVVV.transpose(0,3,2,1)
        v1b -= numpy.einsum('jabc,jc->ba', OVVV, t1b)
        v5b += einsum('KDBC,JKCD->BJ', OVVV, t2bb) * .5
        wOOVO += einsum('idcb,kjbd->ijck', OVVV, taubb) * .25
        wOVVO += einsum('jbcd,kd->jbck', OVVV, t1b)
        wVVVO -= OVVV.conj().transpose(3,2,1,0) * .5
        wVVVO += einsum('jacd,kjbd->cabk', OVVV, t2bb)
        wVVvo += einsum('JACD,kJbD->CAbk', OVVV, t2ab)
        if checkpoint == 18:
            return dict(
                OVVV=OVVV, t1b=t1b, t1a=t1a,
                v1a=v1a.data, v1b=v1b.data)
        OVVV = tmp = None

    if nvirb > 0 and nocca > 0:
        OVvv = numpy.asarray(eris.get_OVvv())
        v1a += numpy.einsum('JCba,JC->ba', OVvv, t1b)
        v5a += einsum('KDbc,jKcD->bj', OVvv, t2ab)
        wOOvo += einsum('IDcb,kJbD->IJck', OVvv, tauab)
        wOVvo += einsum('JBcd,kd->JBck', OVvv, t1a)
        wOvvO -= einsum('JDcb,KD->JbcK', OVvv, t1b)
        wvvVO -= OVvv.conj().transpose(3,2,1,0)
        wvvvo -= einsum('KDca,jKbD->cabj', OVvv, t2ab)
        wvvVO -= einsum('KDca,JKBD->caBJ', OVvv, t2bb)
        wVVvo += einsum('KAcd,jKdB->BAcj', OVvv, t2ab)
        if checkpoint == 19:
            return dict(w3a=w3a, w3b=w3b, OVvv=OVvv)
        OVvv = tmp = None

    if nvira > 0 and noccb > 0:
        ovVV = numpy.asarray(eris.get_ovVV())
        v1b += numpy.einsum('jcBA,jc->BA', ovVV, t1a)
        v5b += einsum('kdBC,kJdC->BJ', ovVV, t2ab)
        wooVO += einsum('idCB,jKdB->ijCK', ovVV, tauab)
        wovVO += einsum('jbCD,KD->jbCK', ovVV, t1b)
        woVVo -= einsum('jdCB,kd->jBCk', ovVV, t1a)
        wVVvo -= ovVV.conj().transpose(3,2,1,0)
        wVVVO -= einsum('kdCA,kJdB->CABJ', ovVV, t2ab)
        wVVvo -= einsum('kdCA,jkbd->CAbj', ovVV, t2aa)
        wvvVO += einsum('kaCD,kJbD->baCJ', ovVV, t2ab)
        ovVV = tmp = None

    w3a += v5a
    w3b += v5b
    w3a += lib.einsum('cb,jb->cj', v1a, t1a)
    w3b += lib.einsum('cb,jb->cj', v1b, t1b)
    w3a -= lib.einsum('jk,jb->bk', v2a, t1a)
    w3b -= lib.einsum('jk,jb->bk', v2b, t1b)

    if checkpoint == 20:
        return dict(
            v1a=v1a, v1b=v1b, v2a=v2a, v2b=v2b,
            w3a=w3a, w3b=w3b, v5a=v5a, v5b=v5b,
        )


def update_lambda_checkpoint_serial(mycc, t1, t2, l1, l2, eris, imds, checkpoint: int = 5):
    from pyscf.lib import logger, einsum
    import numpy
    from pyscf import lib
    from pyscf.cc import uccsd

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    u1a = numpy.zeros_like(l1a)
    u1b = numpy.zeros_like(l1b)
    u2aa = numpy.zeros_like(l2aa)
    u2ab = numpy.zeros_like(l2ab)
    u2bb = numpy.zeros_like(l2bb)
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + mycc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + mycc.level_shift

    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    v1a = imds.v1a - numpy.diag(mo_ea_v)
    v1b = imds.v1b - numpy.diag(mo_eb_v)
    v2a = imds.v2a - numpy.diag(mo_ea_o)
    v2b = imds.v2b - numpy.diag(mo_eb_o)

    mvv = einsum('klca,klcb->ba', l2aa, t2aa) * .5
    mvv+= einsum('lKaC,lKbC->ba', l2ab, t2ab)
    mVV = einsum('klca,klcb->ba', l2bb, t2bb) * .5
    mVV+= einsum('kLcA,kLcB->BA', l2ab, t2ab)
    moo = einsum('kicd,kjcd->ij', l2aa, t2aa) * .5
    moo+= einsum('iKdC,jKdC->ij', l2ab, t2ab)
    mOO = einsum('kicd,kjcd->ij', l2bb, t2bb) * .5
    mOO+= einsum('kIcD,kJcD->IJ', l2ab, t2ab)

    #m3 = lib.einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5
    m3aa, m3ab, m3bb = mycc._add_vvvv(None, (l2aa.conj(),l2ab.conj(),l2bb.conj()), eris)

    if checkpoint == 4:
        return dict(
            t2aa=t2aa, t2ab=t2ab, t2bb=t2bb,
            l2aa=l2aa, l2ab=l2ab, l2bb=l2bb,
            m3aa=m3aa, m3ab=m3ab, m3bb=m3bb)

    m3aa = m3aa.conj()
    m3ab = m3ab.conj()
    m3bb = m3bb.conj()
    m3aa += lib.einsum('klab,ikjl->ijab', l2aa, numpy.asarray(imds.woooo))
    m3bb += lib.einsum('klab,ikjl->ijab', l2bb, numpy.asarray(imds.wOOOO))
    m3ab += lib.einsum('kLaB,ikJL->iJaB', l2ab, numpy.asarray(imds.wooOO))

    if checkpoint == 5:
        return dict(
            mvv=mvv, mVV=mVV, moo=moo, mOO=mOO,
            m3aa=m3aa, m3ab=m3ab, m3bb=m3bb,
            v2a=v2a)

    ovov = numpy.asarray(eris.ovov)
    ovov = ovov - ovov.transpose(0,3,2,1)
    OVOV = numpy.asarray(eris.OVOV)
    OVOV = OVOV - OVOV.transpose(0,3,2,1)
    ovOV = numpy.asarray(eris.ovOV)
    mvv1 = einsum('jc,jb->bc', l1a, t1a) + mvv
    mVV1 = einsum('jc,jb->bc', l1b, t1b) + mVV
    moo1 = einsum('ic,kc->ik', l1a, t1a) + moo
    mOO1 = einsum('ic,kc->ik', l1b, t1b) + mOO

    if nvira > 0 and nocca > 0:
        ovvv = numpy.asarray(eris.get_ovvv())
        ovvv = ovvv - ovvv.transpose(0,3,2,1)
        tmp = lib.einsum('ijcd,kd->ijck', l2aa, t1a)
        m3aa -= lib.einsum('kbca,ijck->ijab', ovvv, tmp)

        tmp = einsum('ic,jbca->jiba', l1a, ovvv)
        tmp+= einsum('kiab,jk->ijab', l2aa, v2a)
        tmp-= einsum('ik,kajb->ijab', moo1, ovov)
        u2aa += tmp - tmp.transpose(1,0,2,3)
        u1a += numpy.einsum('iacb,bc->ia', ovvv, mvv1)
        ovvv = tmp = None

    if nvirb > 0 and noccb > 0:
        OVVV = numpy.asarray(eris.get_OVVV())
        OVVV = OVVV - OVVV.transpose(0,3,2,1)
        tmp = lib.einsum('ijcd,kd->ijck', l2bb, t1b)
        m3bb -= lib.einsum('kbca,ijck->ijab', OVVV, tmp)

        tmp = einsum('ic,jbca->jiba', l1b, OVVV)
        tmp+= einsum('kiab,jk->ijab', l2bb, v2b)
        tmp-= einsum('ik,kajb->ijab', mOO1, OVOV)
        u2bb += tmp - tmp.transpose(1,0,2,3)
        u1b += numpy.einsum('iaCB,BC->ia', OVVV, mVV1)
        OVVV = tmp = None

    if nvirb > 0 and nocca > 0:
        OVvv = numpy.asarray(eris.get_OVvv())
        tmp = lib.einsum('iJcD,KD->iJcK', l2ab, t1b)
        m3ab -= lib.einsum('KBca,iJcK->iJaB', OVvv, tmp)

        tmp = einsum('ic,JAcb->JibA', l1a, OVvv)
        tmp-= einsum('kIaB,jk->IjaB', l2ab, v2a)
        tmp-= einsum('IK,jaKB->IjaB', mOO1, ovOV)
        u2ab += tmp.transpose(1,0,2,3)
        u1b += numpy.einsum('iacb,bc->ia', OVvv, mvv1)
        OVvv = tmp = None

    if nvira > 0 and noccb > 0:
        ovVV = numpy.asarray(eris.get_ovVV())
        tmp = lib.einsum('iJdC,kd->iJCk', l2ab, t1a)
        m3ab -= lib.einsum('kaCB,iJCk->iJaB', ovVV, tmp)

        tmp = einsum('IC,jbCA->jIbA', l1b, ovVV)
        tmp-= einsum('iKaB,JK->iJaB', l2ab, v2b)
        tmp-= einsum('ik,kaJB->iJaB', moo1, ovOV)
        u2ab += tmp
        u1a += numpy.einsum('iaCB,BC->ia', ovVV, mVV1)
        ovVV = tmp = None

    if checkpoint == 10:
        return dict(
            u2aa=u2aa, u2bb=u2bb, u2ab=u2ab,
            u1a=u1a, u1b=u1b,
        )

    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
    tmp = lib.einsum('ijcd,klcd->ijkl', l2aa, tauaa)
    ovov = numpy.asarray(eris.ovov)
    ovov = ovov - ovov.transpose(0,3,2,1)
    m3aa += lib.einsum('kalb,ijkl->ijab', ovov, tmp) * .25

    tmp = lib.einsum('ijcd,klcd->ijkl', l2bb, taubb)
    OVOV = numpy.asarray(eris.OVOV)
    OVOV = OVOV - OVOV.transpose(0,3,2,1)
    m3bb += lib.einsum('kalb,ijkl->ijab', OVOV, tmp) * .25

    tmp = lib.einsum('iJcD,kLcD->iJkL', l2ab, tauab)
    ovOV = numpy.asarray(eris.ovOV)
    m3ab += lib.einsum('kaLB,iJkL->iJaB', ovOV, tmp) * .5
    tmp = lib.einsum('iJdC,lKdC->iJKl', l2ab, tauab)
    m3ab += lib.einsum('laKB,iJKl->iJaB', ovOV, tmp) * .5

    u1a += numpy.einsum('ijab,jb->ia', m3aa, t1a)
    u1a += numpy.einsum('iJaB,JB->ia', m3ab, t1b)
    u1b += numpy.einsum('IJAB,JB->IA', m3bb, t1b)
    u1b += numpy.einsum('jIbA,jb->IA', m3ab, t1a)

    u2aa += m3aa
    u2bb += m3bb
    u2ab += m3ab
    u2aa += ovov.transpose(0,2,1,3)
    u2bb += OVOV.transpose(0,2,1,3)
    u2ab += ovOV.transpose(0,2,1,3)

    if checkpoint == 15:
        return dict(
            tmp=tmp, ovOV=ovOV,
            m3aa=m3aa, m3ab=m3ab, m3bb=m3bb,
            u2aa=u2aa, u2bb=u2bb, u2ab=u2ab,
            u1a=u1a, u1b=u1b)

    fov1 = fova + numpy.einsum('kcjb,kc->jb', ovov, t1a)
    fov1+= numpy.einsum('jbKC,KC->jb', ovOV, t1b)
    tmp = numpy.einsum('ia,jb->ijab', l1a, fov1)
    tmp+= einsum('kica,jbck->ijab', l2aa, imds.wovvo)
    tmp+= einsum('iKaC,jbCK->ijab', l2ab, imds.wovVO)
    tmp = tmp - tmp.transpose(1,0,2,3)
    u2aa += tmp - tmp.transpose(0,1,3,2)

    fov1 = fovb + numpy.einsum('kcjb,kc->jb', OVOV, t1b)
    fov1+= numpy.einsum('kcJB,kc->JB', ovOV, t1a)
    tmp = numpy.einsum('ia,jb->ijab', l1b, fov1)
    tmp+= einsum('kica,jbck->ijab', l2bb, imds.wOVVO)
    tmp+= einsum('kIcA,JBck->IJAB', l2ab, imds.wOVvo)
    tmp = tmp - tmp.transpose(1,0,2,3)
    u2bb += tmp - tmp.transpose(0,1,3,2)

    fov1 = fovb + numpy.einsum('kcjb,kc->jb', OVOV, t1b)
    fov1+= numpy.einsum('kcJB,kc->JB', ovOV, t1a)
    u2ab += numpy.einsum('ia,JB->iJaB', l1a, fov1)
    u2ab += einsum('iKaC,JBCK->iJaB', l2ab, imds.wOVVO)
    u2ab += einsum('kica,JBck->iJaB', l2aa, imds.wOVvo)
    u2ab += einsum('kIaC,jBCk->jIaB', l2ab, imds.woVVo)
    u2ab += einsum('iKcA,JbcK->iJbA', l2ab, imds.wOvvO)
    fov1 = fova + numpy.einsum('kcjb,kc->jb', ovov, t1a)
    fov1+= numpy.einsum('jbKC,KC->jb', ovOV, t1b)
    u2ab += numpy.einsum('ia,jb->jiba', l1b, fov1)
    u2ab += einsum('kIcA,jbck->jIbA', l2ab, imds.wovvo)
    u2ab += einsum('KICA,jbCK->jIbA', l2bb, imds.wovVO)

    ovoo = numpy.asarray(eris.ovoo)
    ovoo = ovoo - ovoo.transpose(2,1,0,3)
    OVOO = numpy.asarray(eris.OVOO)
    OVOO = OVOO - OVOO.transpose(2,1,0,3)
    OVoo = numpy.asarray(eris.OVoo)
    ovOO = numpy.asarray(eris.ovOO)
    tmp = einsum('ka,jbik->ijab', l1a, ovoo)
    tmp+= einsum('ijca,cb->ijab', l2aa, v1a)
    tmp+= einsum('ca,icjb->ijab', mvv1, ovov)
    u2aa -= tmp - tmp.transpose(0, 1, 3, 2)
    tmp = einsum('ka,jbik->ijab', l1b, OVOO)
    tmp+= einsum('ijca,cb->ijab', l2bb, v1b)
    tmp+= einsum('ca,icjb->ijab', mVV1, OVOV)
    u2bb -= tmp - tmp.transpose(0, 1, 3, 2)
    u2ab -= einsum('ka,JBik->iJaB', l1a, OVoo)
    u2ab += einsum('iJaC,CB->iJaB', l2ab, v1b)
    u2ab -= einsum('ca,icJB->iJaB', mvv1, ovOV)
    u2ab -= einsum('KA,ibJK->iJbA', l1b, ovOO)
    u2ab += einsum('iJcA,cb->iJbA', l2ab, v1a)
    u2ab -= einsum('CA,ibJC->iJbA', mVV1, ovOV)

    if checkpoint == 20:
        return dict(
            v1a=v1a, v1b=v1b,
            mvv1=mvv1, mVV1=mVV1,
            u2aa=u2aa, u2bb=u2bb, u2ab=u2ab,
            u1a=u1a, u1b=u1b)

    u1a += fova
    u1b += fovb
    u1a += einsum('ib,ba->ia', l1a, v1a)
    u1a -= einsum('ja,ij->ia', l1a, v2a)
    u1b += einsum('ib,ba->ia', l1b, v1b)
    u1b -= einsum('ja,ij->ia', l1b, v2b)

    u1a += numpy.einsum('jb,iabj->ia', l1a, eris.ovvo)
    u1a -= numpy.einsum('jb,ijba->ia', l1a, eris.oovv)
    u1a += numpy.einsum('JB,iaBJ->ia', l1b, eris.ovVO)
    u1b += numpy.einsum('jb,iabj->ia', l1b, eris.OVVO)
    u1b -= numpy.einsum('jb,ijba->ia', l1b, eris.OOVV)
    u1b += numpy.einsum('jb,iabj->ia', l1a, eris.OVvo)

    u1a -= einsum('kjca,ijck->ia', l2aa, imds.woovo)
    u1a -= einsum('jKaC,ijCK->ia', l2ab, imds.wooVO)
    u1b -= einsum('kjca,ijck->ia', l2bb, imds.wOOVO)
    u1b -= einsum('kJcA,IJck->IA', l2ab, imds.wOOvo)

    u1a -= einsum('ikbc,back->ia', l2aa, imds.wvvvo)
    u1a -= einsum('iKbC,baCK->ia', l2ab, imds.wvvVO)
    u1b -= einsum('IKBC,BACK->IA', l2bb, imds.wVVVO)
    u1b -= einsum('kIcB,BAck->IA', l2ab, imds.wVVvo)

    u1a += numpy.einsum('jiba,bj->ia', l2aa, imds.w3a)
    u1a += numpy.einsum('iJaB,BJ->ia', l2ab, imds.w3b)
    u1b += numpy.einsum('JIBA,BJ->IA', l2bb, imds.w3b)
    u1b += numpy.einsum('jIbA,bj->IA', l2ab, imds.w3a)

    tmpa  = t1a + numpy.einsum('kc,kjcb->jb', l1a, t2aa)
    tmpa += numpy.einsum('KC,jKbC->jb', l1b, t2ab)
    tmpa -= einsum('bd,jd->jb', mvv1, t1a)
    tmpa -= einsum('lj,lb->jb', moo, t1a)
    tmpb  = t1b + numpy.einsum('kc,kjcb->jb', l1b, t2bb)
    tmpb += numpy.einsum('kc,kJcB->JB', l1a, t2ab)
    tmpb -= einsum('bd,jd->jb', mVV1, t1b)
    tmpb -= einsum('lj,lb->jb', mOO, t1b)

    if checkpoint == 25:
        return dict(
            woovo=imds.woovo,
            wooVO=imds.wooVO,
            wOOVO=imds.wOOVO,
            wOOvo=imds.wOOvo,
            wvvvo=imds.wvvvo,
            tmpa=tmpa, tmpb=tmpb,
            w3a=imds.w3a, w3b=imds.w3b,
            v1a=v1a, v1b=v1b, v2a=v2a, v2b=v2b,
            u2aa=u2aa, u2bb=u2bb, u2ab=u2ab,
            u1a=u1a, u1b=u1b)

    u1a += numpy.einsum('jbia,jb->ia', ovov, tmpa)
    u1a += numpy.einsum('iaJB,JB->ia', ovOV, tmpb)
    u1b += numpy.einsum('jbia,jb->ia', OVOV, tmpb)
    u1b += numpy.einsum('jbIA,jb->IA', ovOV, tmpa)

    u1a -= numpy.einsum('iajk,kj->ia', ovoo, moo1)
    u1a -= numpy.einsum('iaJK,KJ->ia', ovOO, mOO1)
    u1b -= numpy.einsum('iajk,kj->ia', OVOO, mOO1)
    u1b -= numpy.einsum('IAjk,kj->IA', OVoo, moo1)

    if checkpoint == 30:
        return dict(
            u2aa=u2aa, u2bb=u2bb, u2ab=u2ab,
            u1a=u1a, u1b=u1b)

    tmp  = fova - numpy.einsum('kbja,jb->ka', ovov, t1a)
    tmp += numpy.einsum('kaJB,JB->ka', ovOV, t1b)
    u1a -= lib.einsum('ik,ka->ia', moo, tmp)
    u1a -= lib.einsum('ca,ic->ia', mvv, tmp)
    tmp  = fovb - numpy.einsum('kbja,jb->ka', OVOV, t1b)
    tmp += numpy.einsum('jbKA,jb->KA', ovOV, t1a)
    u1b -= lib.einsum('ik,ka->ia', mOO, tmp)
    u1b -= lib.einsum('ca,ic->ia', mVV, tmp)

    eia = lib.direct_sum('i-j->ij', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-j->ij', mo_eb_o, mo_eb_v)

    if checkpoint == 35:
        return dict(
            tmp=tmp, mOO=mOO, mVV=mVV,
            u2aa=u2aa, u2bb=u2bb, u2ab=u2ab,
            u1a=u1a, u1b=u1b,
            eia=eia, eIA=eIA,
            )

    u1a /= eia
    u1b /= eIA

    u2aa /= lib.direct_sum('ia+jb->ijab', eia, eia)
    u2ab /= lib.direct_sum('ia+jb->ijab', eia, eIA)
    u2bb /= lib.direct_sum('ia+jb->ijab', eIA, eIA)

    if checkpoint == 36:
        return dict(
            u1a=u1a, u1b=u1b,
            u2aa=u2aa, u2ab=u2ab, u2bb=u2bb
        )

    time0 = log.timer_debug1('update l1 l2', *time0)
    return (u1a,u1b), (u2aa,u2ab,u2bb)


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

    # pr = cProfile.Profile()
    # pr.enable()

    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t1, t2 = cc.init_amps()
    oovvs = uccsd_mpi._add_vvvv(t1=t1, t2=t2, eris=cc._eris)

    # pr.disable()
    # pr.dump_stats(f'uccsd_mpi_add_vvvv_rank{rank}.prof')
    keys = ('oovv_aa', 'oovv_ab', 'oovv_bb')
    res = {k: cc_tools.collect_array(oovvs[i], seg_idx=2) for i, k in enumerate(keys)}
    return res


@mpi.parallel_call
def test_update_amps_checkpoint(cc: UCCSD_MPI, checkpoint: int = 35):
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t1, t2 = cc.init_amps()
    res = cc.update_amps(t1, t2, cc._eris, checkpoint=checkpoint)
    return res


@mpi.parallel_call
def test_update_amps(cc: UCCSD_MPI):
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t1, t2 = cc.init_amps()
    for _ in range(3):
        t1, t2 = cc.update_amps(t1, t2, cc._eris)
    (t1a, t1b), (t2aa, t2ab, t2bb) = t1, t2
    res = dict(t1a=t1a, t1b=t1b)
    for k, v in zip(('t2aa', 't2ab', 't2bb'), (t2aa, t2ab, t2bb)):
        res[k] = cc_tools.collect_array(v, seg_idx=2)
    return res


@mpi.parallel_call
def test_energy(cc: UCCSD_MPI):
    max_cycle = 10
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t1, t2 = cc.init_amps()
    for _ in range(max_cycle):
        t1, t2 = cc.update_amps(t1, t2, cc._eris)
    E = cc.energy(t1=t1, t2=t2, eris=cc._eris)
    return E


@mpi.parallel_call
def test_lambda_intermediates_checkpoint(cc: UCCSD_MPI, checkpoint: int = 10):
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t1, t2 = cc.init_amps()
    res = uccsd_lambda_mpi.make_intermediates(cc, t1, t2, cc._eris, checkpoint=checkpoint)
    return res


@mpi.parallel_call
def test_lambda_intermediates(cc: UCCSD_MPI):
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t1, t2 = cc.init_amps()
    imds = uccsd_lambda_mpi.make_intermediates(cc, t1, t2, cc._eris)
    return {k.replace('x', 'v').replace('X', 'V'): getattr(imds, k).collect().data
            for k in imds._keys}


@mpi.parallel_call
def test_update_lambda_checkpoint(cc: UCCSD_MPI, checkpoint: int = 5):
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t1, t2 = cc.init_amps()
    l1, l2 = t1, t2
    imds = uccsd_lambda_mpi.make_intermediates(cc, t1, t2, cc._eris)
    res = uccsd_lambda_mpi.update_lambda(
        cc, t1, t2, l1, l2, cc._eris, imds, checkpoint=checkpoint)
    imds = {k.replace('x', 'v').replace('X', 'V'): getattr(imds, k).collect().data
            for k in imds._keys}
    return res, imds


@mpi.parallel_call
def test_update_lambda(cc: UCCSD_MPI):
    return_segarray = True
    if getattr(cc, '_eris', None) is None:
        cc.ao2mo(cc.mo_coeff)
    _, t10, t20 = cc.init_amps()
    t1, t2 = cc.update_amps(t10, t20, cc._eris)
    t1, t2 = cc.update_amps(t1, t2, cc._eris)
    l1, l2 = t10, t20
    imds = uccsd_lambda_mpi.make_intermediates(cc, t1, t2, cc._eris)
    l1new, l2new = uccsd_lambda_mpi.update_lambda(
        cc, t1, t2, l1, l2, cc._eris, imds, checkpoint=None, return_segarray=return_segarray)
    (l1a, l1b), (l2aa, l2ab, l2bb) = l1new, l2new
    if return_segarray:
        res = dict(l1a=l1a.data, l1b=l1b.data,
                l2aa=l2aa.collect().data,
                l2ab=l2ab.collect().data,
                l2bb=l2bb.collect().data)
    else:
        res = dict(l1a=l1a, l1b=l1b)
        for k, v in zip(('l2aa', 'l2ab', 'l2bb'), (l2aa, l2ab, l2bb)):
            res[k] = cc_tools.collect_array(v, seg_idx=2)
    return res


@mpi.parallel_call
def test_rdm_single_shot():
    """"""
