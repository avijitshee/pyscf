from pyscf import lib, df
import os

from numpy.linalg import qr
from pyscf import lib, df
from pyscf.lib import logger
import numpy as np
from dataclasses import dataclass

from pyscf.sylvester_cc import mpcc_tools
from pyscf.sylvester_cc import laplace_quadrature
from functools import partial

import time 

class MPCC_LL:
    def __init__(self, mf, eris, frags, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        if 'll_max_its' in kwargs:
            self.ll_max_its = kwargs['ll_max_its']
        else:    
            self.ll_max_its = 50

        if 'll_con_tol' in kwargs:
            self.ll_con_tol = kwargs['ll_con_tol']
        else:
            self.ll_con_tol = 1e-6
 
        if 'll_kernel_type' in kwargs:
            self.kernel_type = kwargs['ll_kernel_type']
        else:
            self.kernel_type = 'factorized'

        if 'll_method' in kwargs:
            self.ll_method = kwargs['ll_method']
        else:
            self.ll_method = 'T1_transform'

        if 'll_low_rank_tol' in kwargs:
            self.ll_low_rank_tol = kwargs['ll_low_rank_tol']
        else:
            self.ll_low_rank_tol = None

        self.ll_laplace_quad = kwargs.get('ll_laplace_quad', None)
        self.ll_laplace_quad_file = kwargs.get('ll_laplace_quad_file', None)
        self.ll_laplace_root = kwargs.get('ll_laplace_root', None)
        self.ll_laplace_npoints = kwargs.get(
            'll_laplace_npoints', kwargs.get('ll_laplace_nlap', 16)
        )
        self.ll_active_t2_tol = kwargs.get(
            'll_active_t2_tol', min(self.ll_con_tol, 1.0e-6)
        )
        self.ll_active_t2_max_its = kwargs.get('ll_active_t2_max_its', 1000)

        self._kernels = {
                'factorized': self._factorized_kernel,
                'unfactorized': self._unfactorized_kernel, 
                'sylvester_laplace_factorized': self._sylvester_laplace_factorized_noniterative_kernel,
                }

        self.frags = frags

        #NOTE can be potentially initialized
        #self.t1 = None
        self.t2 = None
        self._Y = None
        self._t2_full = None

        # NOTE use DIIS as default
        self.diis = True

        self._eris = eris
        self._e_corr = None

    @property
    def e_tot(self):
        if self._e_corr is None:
            print('MPCC did not run, return mean field solution:')
            return float(self.mf.e_tot)
        else:
            return float(self.mf.e_tot + self._e_corr)

    @property
    def e_corr(self):
        return float(self._e_corr)

    @property
    def nvir(self):
        return self.mf.mol.nao - self.nocc
    
    @property
    def nocc(self):
        return self.mf.mol.nelec[0]


def get_X(self, t1):

    t1a,t1b = t1
    Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1a)
    Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1a)

    XVO = lib.einsum("Lab,ib->Lai", self._eris.LVV, t1b)
    XOO = lib.einsum("Lia,ja->Lij", self._eris.LOV, t1b)

    X = lib.einsum("Lia,ia->L", self._eris.Lov, t1a) + lib.einsum("Lia,ia->L", self._eris.LOV, t1b)

    return Xoo, Xvo, X, XVO, XOO

def add_t2_to_fock(self, Fvv, Foo, Xvo_t2):    

    Foo += lib.einsum("Lie,Lej->ij", self._eris.Lov,Xvo_t2)
    Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo_t2)

    return Foo, Fvv


def get_Xvo_t2(self, t2):

    t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)

    Xvo_t2 = lib.einsum("Lkc,ikac ->Lai", self._eris.Lov, t2_antisym)

    return Xvo_t2

def get_J(self, Xoo, Xvo, XOO, XVO, t1):

    Joo = Xoo + self._eris.Loo
    Jvo = (
        Xvo + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
    )
    Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1a)

    Jov = self._eris.Lov

    JOO = XOO + self._eris.LOO
    JVO = (
        XVO + self._eris.LVO - lib.einsum("Lji,ja->Lai", JOO, t1b)
    )

    JVV = self._eris.LVV - lib.einsum("Lkb,ka->Lab", self._eris.LOV, t1b)
    JOV = self._eris.LOV

    return Joo, Jvo, JOO, JVO, JVV, JOV

def get_F(self, t1, X, Xoo, Xvo):

    Foo = self._eris.foo.copy()
    Foo += lib.einsum("Lij,L->ij", self._eris.Loo, X)
    Foo -= lib.einsum("Lmj,Lim->ij", self._eris.Loo,Xoo)

    Fvv = self._eris.fvv.copy()
    Fvv += lib.einsum("Lab,L->ab",self._eris.Lvv,X) 
    Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo)

    Fov  = self._eris.fov.copy()
    Fov += lib.einsum("Ljb,L->jb", self._eris.Lov, X)
    Fov -= lib.einsum("Lji,Lib->jb", Xoo, self._eris.Lov)

    FOO = self._eris.fockb[:noccb,:noccb].copy()
    FOO += lib.einsum("Lij,L->ij", self._eris.LOO, X)
    FOO -= lib.einsum("Lmj,Lim->ij", self._eris.LOO,XOO)

    FVV = self._eris.fockb[noccb:,noccb:].copy()
    FVV += lib.einsum("Lab,L->ab",self._eris.LVV,X)
    FVV -= lib.einsum("Lmb,Lam->ab",self._eris.LOV,XVO)

    FOV  = self._eris.fockb[:noccb,noccb:].copy()
    FOV += lib.einsum("Ljb,L->jb", self._eris.LOV, X)
    FOV -= lib.einsum("Lji,Lib->jb", XOO, self._eris.LOV)

    return Foo, Fvv, Fov