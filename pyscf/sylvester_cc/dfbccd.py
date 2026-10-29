from pyscf import lib, df
from pyscf.lib import logger
import numpy as np
from dataclasses import dataclass


class DFRCCSD:
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

        Joo, Jvo, Jvv = self.get_J(Xoo, Xvo, Xvo_t2, t1)
        
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo, Jvo)

        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Xvo_t2)    

        Ω = self.get_Ω(X, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2)
        res2 = self.update_t2(t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv)

        #make the the inactive residuals zero

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

    def get_J(self, Xoo, Xvo, Xvo_t2, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + Xvo_t2 + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
        )

        Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) #we don't need this here  

        return Joo, Jvo, Jvv


    def get_F(self, t1, X, Xoo, Xvo, Jvo):

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


    def _ppl_contraction_symmetric(self, factors, t2):
        """Evaluate the active-space PPL term in dense packed-pair form.

        The symmetric and antisymmetric combinations follow Eqs. (38)-(43) of
        J. Chem. Theory Comput. 2021, 17, 4799-4822.  All packed pair matrices
        are constructed and contracted at once because the high-level active
        space is expected to be small.
        """
        nhole = t2.shape[0]
        nparticle = factors.shape[2]
        nout = factors.shape[1]
        dtype = np.result_type(factors, t2)
        result_shape = (nhole, nhole, nout, nout)
        if nhole == 0 or nparticle == 0 or nout == 0:
            return np.zeros(result_shape, dtype=dtype)

        hole_i, hole_j = np.tril_indices(nhole)
        out_a, out_b = np.tril_indices(nout)
        part_e, part_f = np.tril_indices(nparticle)

        direct = t2[
            hole_i[:, None],
            hole_j[:, None],
            part_e[None, :],
            part_f[None, :],
        ]
        exchange = t2[
            hole_i[:, None],
            hole_j[:, None],
            part_f[None, :],
            part_e[None, :],
        ]
        t_minus = direct - exchange
        off_diagonal_ef = part_e != part_f
        exchange[:, off_diagonal_ef] += direct[:, off_diagonal_ef]
        exchange[:, ~off_diagonal_ef] = direct[:, ~off_diagonal_ef]
        t_plus = exchange
        del direct

        # Assemble all packed interactions in fixed-width panels.  The panels
        # are used to expose large matrix products to BLAS; they are not a
        # low-memory fallback, and the complete V+ and V- matrices remain live.
        npair_out = len(out_a)
        npair_particle = len(part_e)
        v_plus = np.empty((npair_out, npair_particle), dtype=dtype)
        v_minus = np.empty_like(v_plus)
        naux = factors.shape[0]
        itemsize = np.dtype(dtype).itemsize
        # cache constraint: left_panel (panel_size × nparticle × naux) targets ~16 MB L3
        _panel_cache = max(1, (16 << 20) // max(1, nparticle * naux * itemsize))
        # memory constraint: interaction tensor (panel_size × nparticle × nout × nparticle) targets ~256 MB
        _panel_mem = max(1, (256 << 20) // max(1, nparticle * nout * nparticle * itemsize))
        # BLAS floor: keep M-dimension (panel_size × nparticle) >= 64
        _min_panel = max(1, -(-64 // max(1, nparticle)))
        panel_size = max(_min_panel, min(nout, _panel_cache, _panel_mem))
        for a_start in range(0, nout, panel_size):
            a_stop = min(nout, a_start + panel_size)
            left_panel = factors[:, a_start:a_stop, :].reshape(
                factors.shape[0], -1
            )
            right_panel = factors[:, :a_stop, :].reshape(
                factors.shape[0], -1
            )
            interaction = lib.dot(left_panel.T, right_panel).reshape(
                a_stop - a_start,
                nparticle,
                a_stop,
                nparticle,
            )

            for a in range(a_start, a_stop):
                pair_start = a * (a + 1) // 2
                pair_stop = (a + 1) * (a + 2) // 2
                row = interaction[a - a_start, :, : a + 1, :].transpose(
                    1, 0, 2
                )
                direct = row[:, part_e, part_f]
                exchange = row[:, part_f, part_e]
                v_plus[pair_start:pair_stop] = direct + exchange
                v_minus[pair_start:pair_stop] = direct - exchange

            del left_panel, right_panel, interaction

        sigma_plus = lib.dot(t_plus, v_plus.T)
        sigma_plus *= 0.5
        sigma_minus = lib.dot(t_minus, v_minus.T)
        sigma_minus *= 0.5
        del t_plus, t_minus, v_plus, v_minus

        residual_ab = sigma_plus + sigma_minus
        residual_ba = sigma_plus - sigma_minus
        result = np.zeros(result_shape, dtype=dtype)
        result[
            hole_i[:, None],
            hole_j[:, None],
            out_a[None, :],
            out_b[None, :],
        ] = residual_ab

        off_diagonal_ab = out_a != out_b
        if np.any(off_diagonal_ab):
            result[
                hole_i[:, None],
                hole_j[:, None],
                out_b[None, off_diagonal_ab],
                out_a[None, off_diagonal_ab],
            ] = residual_ba[:, off_diagonal_ab]

        off_diagonal_ij = hole_i != hole_j
        if np.any(off_diagonal_ij):
            result[
                hole_j[off_diagonal_ij, None],
                hole_i[off_diagonal_ij, None],
                out_b[None, :],
                out_a[None, :],
            ] = residual_ab[off_diagonal_ij]
            if np.any(off_diagonal_ab):
                result[
                    hole_j[off_diagonal_ij, None],
                    hole_i[off_diagonal_ij, None],
                    out_a[None, off_diagonal_ab],
                    out_b[None, off_diagonal_ab],
                ] = residual_ba[off_diagonal_ij][:, off_diagonal_ab]

        return result


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

    def update_t2(self, t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv):

        Imbje, Imbej, Imnij = self.t2_transform_quadratic(t2)  

        res2 = lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

        #PPL
        #Waebf = lib.einsum("Lae, Lbf -> abef", Jvv, Jvv)
        #res2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)

        res2 += self._ppl_contraction_symmetric(Jvv, t2)

        #HHL
        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo, Joo) + Imnij
        res2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2) 

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)
        
        tmp  = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("mi,mjab->ijab", Foo_tmp, t2)
## N3V3
        W_jebm = lib.einsum("Lmj, Lbe -> mbje", Joo, Jvv) - Imbje 
        tmp -= lib.einsum("mbje, imae -> ijab", W_jebm, t2) # em should be ii, ia, ai types

        W_jema = lib.einsum("Lmj, Lae -> maje", Joo, Jvv) - Imbje*0.5
        tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2) # em should be ii, ia, ai types

## more DCA like terms:
        tmp -= lib.einsum("mbej, imae -> ijab", Imbej, t2) #should it not be antisym

        res2 += tmp + tmp.transpose(1,0,3,2) 

        del Foo_tmp, Fvv_tmp 
       
        return res2

    def t2_transform_quadratic(self,t2):

        #I_mn^ij 
        Vnemf = lib.einsum("Lne, Lmf -> nmef", self._eris.Lov, self._eris.Lov)
        Imnij = lib.einsum("mnef, ijef -> mnij", Vnemf, t2) #ef should be ii, ia, ai types
        #I^je_bm
        Imbej = lib.einsum("nmef, jnbf -> mbej", Vnemf, t2)
        #I^je_mb
        Imbje = lib.einsum("nmef, jnfb -> mbje", Vnemf, t2) # #nf should be ii, ia, ai types

        return Imbje, Imbej, Imnij

    def init_amps(self):
        t2 = lib.einsum("Lai,Lbj->ijab", self._eris.Lvo, self._eris.Lvo)
        t2 /= self._eris.D          
        t1 = self._eris.fov/self._eris.eia
        return t1, t2


    def get_energy_orbital_optimized(self, t1, L1, t2, Jov):
       """
       Calculate total energy using the kappa, T and L amplitudes.
       step1: evaluate 1-RDM using the kappa amplitudes.
       step2: evaluate the energy using the 1-RDM and the integrals. Construct the Fock matrix using the 1-RDM and the integrals,
         and then take the trace with the 1-RDM.
       step3: construct the T1 transformed integrals, which are completely contractable with the T2 amplitudes
         using the kappa amplitudes, and then evaluate the energy.  
       """
       #step 1:
       # we can use the same renormalization procedure as in the amplitude updates to get the renormalized kappa amplitudes, and then use them to evaluate the 1-RDM.
       t1_renorm, L1_renorm, den_oo, den_vv = self.get_renormalized_t1(t1, L1)
       #evaluate the 1-RDM using the renormalized kappa amplitudes
       doo = lib.einsum("ia,ja->ij", t1_renorm, L1_renorm) #check sign
       dvv = - lib.einsum("ia,ib->ab", t1_renorm, L1_renorm)
       dvo = t1_renorm.T - lib.einsum('im,ma->ai', doo, t1_renorm) #check sign
       dov = L1_renorm + lib.einsum('ca,ic->ia', dvv, L1_renorm)
       #collect the 1-RDM
       one_rdm = np.zeros((self.nocc + self.nvir, self.nocc + self.nvir))
       occ_idx = np.arange(self.nocc)
       vir_idx = np.arange(self.nocc, self.nocc + self.nvir)
       one_rdm[np.ix_(occ_idx, occ_idx)] = 1.0 * doo
       one_rdm[np.ix_(vir_idx, vir_idx)] = 1.0 * dvv
       one_rdm[np.ix_(vir_idx, occ_idx)] = dvo
       one_rdm[np.ix_(occ_idx, vir_idx)] = dov

       #print diagonal elements of the 1-RDM
       print("Diagonal elements of the 1-RDM:")
       print(np.diag(one_rdm))

       #collect all integrals into a supermatrix Lpq, where p and q run over all orbitals.

       Lpq = np.zeros((self.naux,self.nocc + self.nvir, self.nocc + self.nvir))

       naux_idx = np.arange(self.naux)

       Lpq[np.ix_(naux_idx, occ_idx, occ_idx)] = self._eris.Loo
       Lpq[np.ix_(naux_idx, vir_idx, vir_idx)] = self._eris.Lvv
       Lpq[np.ix_(naux_idx, occ_idx, vir_idx)] = self._eris.Lov
       Lpq[np.ix_(naux_idx, vir_idx, occ_idx)] = self._eris.Lvo

      #X = lib.einsum('Lrs,rs->L', Lpq, one_rdm)
      #Fock = lib.einsum('Lpq,X->pq', Lpq, X)

       #construct the Fock matrix:

       Fock = 2.0*np.einsum('Lpr,Lqs,qs->pq', Lpq, Lpq, one_rdm, optimize=True)
       Fock -= np.einsum('Lps,Lqr,qs->pq', Lpq, Lpq, one_rdm, optimize=True)

       e_1b = lib.einsum('pq,qp->', Fock, one_rdm)

       print("One-body contribution to the energy:", e_1b)

       #step 2: construct the T1 transformed integrals, which are completely contractable with the T2 amplitudes
       ovov_t1 = lib.einsum("Lia,Ljb->iajb", Jov, Jov)

       e_2b = 2*lib.einsum('ijab,iajb', t2, ovov_t1)
       e_2b -=  lib.einsum('ijab,ibja', t2, ovov_t1)
       e_corr = e_1b + e_2b
       return e_corr

    
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
