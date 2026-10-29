from pyscf import lib, df
from pyscf.lib import logger
import numpy as np
from dataclasses import dataclass


class DFRPA:
    def __init__(self, mf, eris, **kwargs):
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

        # NOTE can be potentially initialized
       # self.t1 = None
       # self.t2 = None

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
    def nvir(self):
        return self.mf.mol.nao - self.nocc
    
    @property
    def nocc(self):
        return self.mf.mol.nelec[0]

    def kernel(self, t1=None, t2=None):

        # NOTE Do we want to initialize t1 and t2?

        if t1 is None and t2 is None:
            t1, t2 = self.init_amps()
           
        err = np.inf
        count = 0
        adiis = lib.diis.DIIS()
        adiis.space = 8

        e_corr = None
        while err > self.ll_con_tol and count < self.ll_max_its:

            res, t1_new, t2_new = self.updated_amps(t1, t2)
            if self.diis:
                t1_new, t2_new = self.run_diis(t1_new, t2_new, adiis)
            else:
                t1_new, t2_new = t1_new, t2_new

            t1, t2 = t1_new, t2_new
            t1_new, t2_new = None, None  # free memory
            
            count += 1
            err = res
            # NOTE change this to logger!
            print(f"It {count}; residual {res:.6e}")

        return t1, t2

    def updated_amps(self, t1, t2):
        """
        Following Table XXX in Future Paper
        """

        # Contractions
        X, Xoo, Xvo, Xvo_t2 = self.get_X(t1, t2)

        Joo, Jvo = self.get_J(Xoo, Xvo, Xvo_t2, t1, t2)
        
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Xvo_t2)    

        Ω = self.get_Ω(X, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2)
        res2 = self.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)

        res1 = Ω.T / self._eris.eia
        res2 = res2 / self._eris.D

        res = np.linalg.norm(res1) + np.linalg.norm(res2)

        t1 -= res1
        t2 -= res2
        return res, t1, t2
    
    def run_diis(self, t1, t2, adiis):

        vec = self.amplitudes_to_vector(t1, t2)
        t1, t2 = self.vector_to_amplitudes(adiis.update(vec))

        return t1, t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        nov = self.nocc * self.nvir
        size = nov + nov * (nov + 1) // 2
        vector = np.ndarray(size, t1.dtype, buffer=out)
        vector[:nov] = t1.ravel()
        lib.pack_tril(t2.transpose(0, 2, 1, 3).reshape(nov, nov), out=vector[nov:])
        return vector

    def vector_to_amplitudes(self, vector):
        nov = self.nocc * self.nvir
        t1 = vector[:nov].copy().reshape((self.nocc, self.nvir))
        # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
        t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
        t2 = t2.reshape(self.nocc, self.nvir, self.nocc, self.nvir).transpose(
            0, 2, 1, 3
        )
        return t1, np.asarray(t2, order="C")

    def get_X(self, t1, t2):

        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)

        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xvo_t2 = lib.einsum("Lkc,ikac ->Lai", self._eris.Lov, t2_antisym)
        
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = lib.einsum("Lia,ia->L", self._eris.Lov, t1)*2.0

        return X, Xoo, Xvo, Xvo_t2

    def get_J(self, Xoo, Xvo, Xvo_t2, t1, t2):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + Xvo_t2 + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
        )

        #Jvo_ex = -lib.einsum("Lkc,ikca->Lai", self._eris.Lov, t2)

        #Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) #we don't need this here  

        return Joo, Jvo

    def get_F(self, t1, X, Xoo, Xvo):

        nocc, nvir = t1.shape
        Foo = self._eris.foo.copy()
        Foo += lib.einsum("Lij,L->ij", self._eris.Loo, X)
        Foo -= lib.einsum("Lmj,Lim->ij", self._eris.Loo,Xoo)

        Fvv = self._eris.fvv.copy()
        Fvv += lib.einsum("Lab,L->ab",self._eris.Lvv,X) 
        Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo)
              
        Fov  = self._eris.fov.copy()
        Fov += lib.einsum("Ljb,L->jb", self._eris.Lov, X)
        Fov -= lib.einsum("Lji,Lib->jb", Xoo, self._eris.Lov)

        return Foo, Fvv, Fov

    def add_t2_to_fock(self, Fvv, Foo, Xvo_t2):    

        Foo += lib.einsum("Lie,Lej->ij", self._eris.Lov,Xvo_t2)
        Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo_t2)

        return Foo, Fvv


    def get_Ω(self, X, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += 0.5*lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= 0.5*lib.einsum("lb,la->ab",Fov,t1) 

        Ω = self._eris.fov.T.copy()

        Ω -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω -= lib.einsum("Laj,Lji->ai", Xvo_t2, self._eris.Loo)
        Ω += lib.einsum("Lai,L->ai", self._eris.Lvo, X)

        Ω += lib.einsum("Lae,Lei->ai", self._eris.Lvv, Xvo_t2)

        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)

        t2_antisym = 2.0*t2 - np.transpose(t2, (0, 1, 3, 2))
        Ω += lib.einsum("ijab,jb->ai", t2_antisym, Fov)
   
        del Foo_tmp, Fvv_tmp

        return Ω

    def update_t2(self, t2, Jvo, Foo, Fvv, Fov, t1):
        res2 = lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)
        
        tmp  = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("mi,mjab->ijab", Foo_tmp, t2)
## N3V3
        #W_jebm = lib.einsum("Lmj, Lbe -> mbje", Joo, Jvv) #- Imbje 
        #tmp -= lib.einsum("mbje, imae -> ijab", W_jebm, t2) 

        #tmp += lib.einsum("Lai,Lbj->ijab", Jvo_ex, self._eris.Lvo) 

        #tmp -= lib.einsum("mbej, imae -> ijab", Imbej, t2) 
        #tmp -= lib.einsum("Lai,Lbj->ijab", Jvo_ex, Jvo_ex)
        res2 += tmp + tmp.transpose(1,0,3,2) 

        del Foo_tmp, Fvv_tmp 
       
        return res2

    def t2_transform_quadratic(self,t2):
     
        #I_mn^ij 
        Vnemf = lib.einsum("Lne, Lmf -> nmef", self._eris.Lov, self._eris.Lov)
        #I^je_bm
        Imbej = lib.einsum("nmef, jnbf -> mbej", Vnemf, t2)

        return Imbej

    def init_amps(self):
        t2 = lib.einsum("Lai,Lbj->ijab", self._eris.Lvo, self._eris.Lvo)
        t2 /= self._eris.D          
        t1 = self._eris.fov/self._eris.eia
        return t1, t2
    
    def energy(self, t1, t2):                                                     
       '''RCCSD correlation energy'''                                                               
       
       fock = self._eris.fov.copy()
       e = 2*lib.einsum('ia,ia', fock, t1)    
       tau = lib.einsum('ia,jb->ijab',t1,t1)                                                         
       tau += t2 
       eris_ovov = lib.einsum("Lia,Ljb->iajb", self._eris.Lov, self._eris.Lov)
       e += 2*lib.einsum('ijab,iajb', tau, eris_ovov)                                                
       e -=  lib.einsum('ijab,ibja', tau, eris_ovov)                                                
       #if abs(e.imag) > 1e-4:
       #    logger.warn(cc, 'Non-zero imaginary part found in RCCSD energy %s', e)                   
       return e.real                       
