import numpy as np
from pyscf import lib, df
from pyscf.ao2mo import _ao2mo

from pyscf.sylvester_cc import mpcc_tools

class ERIs:

    def __init__(self, mf, mo_coeff=None):

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self.nao = mf.mol.nao
        self.nocc = mf.mol.nelec[0]
        self.nvir = mf.mol.nao - self.nocc
        self.naux = self.with_df.get_naoaux()
        if mo_coeff is None:
           self.mo_coeff = mf.mo_coeff
        else:
           self.mo_coeff = mo_coeff       
        self.mo_energy = mf.mo_energy

        fock_mo = self.mo_coeff.T @ mf.get_fock() @ self.mo_coeff
        self.foo = fock_mo[: self.nocc, : self.nocc]
        self.fvv = fock_mo[self.nocc : self.nao, self.nocc : self.nao]
        self.fov = fock_mo[: self.nocc, self.nocc : self.nao]

        # NOTE change convetion to virtual, occupied 
        self.eia = lib.direct_sum(
            "-i+a->ia", np.diag(self.foo), np.diag(self.fvv)
        )

        # NOTE Shall we keep this call here?
        self._make_df_eris()
        self._make_df_denominator()

    def _make_df_denominator(self):

        self.dD = mpcc_tools.piv_chol_tensor(self.eia)
        
        # NOTE REMOVE THIS LATER!!!
        self.D = lib.direct_sum("ia+jb->ijab", self.eia, self.eia)

          
    def _make_df_eris(self):
        
        Loo = np.empty((self.naux, self.nocc, self.nocc))
        Lov = np.empty((self.naux, self.nocc, self.nvir))
        Lvv = np.empty((self.naux, self.nvir, self.nvir))
        mo = np.asarray(self.mo_coeff, order='F')
        ijslice = (0, self.nao, 0, self.nao)
        p1 = 0
        Lpq = None
        for k, eri1 in enumerate(self.with_df.loop()):
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Lpq = Lpq.reshape(p1 - p0, self.nao, self.nao)
            Loo[p0:p1] = Lpq[:, :self.nocc, :self.nocc]
            Lov[p0:p1] = Lpq[:, :self.nocc, self.nocc:]
            Lvv[p0:p1] = Lpq[:, self.nocc:, self.nocc:]
        Lpq = None
        Lvo = Lov.transpose(0,2,1).reshape(self.naux,self.nvir,self.nocc)
        self.Loo = Loo
        self.Lov = Lov
        self.Lvo = Lvo
        self.Lvv = Lvv       

        # NOTE 06/04 debate about cache 

    # hold all a-a, a-i, i-i tensor segments here 
    # compute Lvv on the fly *lazy?

    # NOTE Remove after discussion on 06/11
    # NOTE move contractions into the "class"
    # NOTE Now comes the second set of equations
    def get_Aia(self, u):

        tmp_eri = np.einsum("Lkc,Lad->kcad", self.Lov, self.Lvv)

        return np.einsum("kicd, kcad -> ia", u, tmp_eri)

    def get_Bia(self, u):

        tmp_eri = np.einsum("Lki,Llc->kilc", self.Loo, self.Lov)

        return np.einsum("klac, -kilc -> ia", u, tmp_eri)

    def get_Cia(self, u):

        return np.einsum("ikac, kc-> ia", u, self.fov)

    def get_Aijab(self, t):

        tmp_eri = np.einsum("Lac, Lbd->acbd", self.Lvv, self.Lvv)

        return np.einsum("ijcd, acbd -> ijab", t, tmp_eri)

    def get_Bijab(self, t):

        tmp_eri = np.einsum("Lkc, Lld -> kcld", self.self.Lov, self.self.Lov)
        B = np.einsum("ijcd, kcld -> ijkl", t, tmp_eri)
        tmp_eri = np.einsum("Lki, Ljl -> ijkl", self.Loo, self.Loo)
        B = tmp_eri + B

        return np.einsum("klab, ijkl -> ijab", t, B)

    def get_Cijab(self, t):

        # NOTE tmp_eri is the same as in get_Bijab
        tmp_eri = np.einsum("Lkc, Lld -> kcld", self.Lov, self.Lov)
        C = np.einsum("liad, kcld -> kiac", t, tmp_eri)
        tmp_eri = np.einsum("Lki, Lac -> kiac", self.Loo, self.Lvv)
        C = tmp_eri - 0.5 * C

        return np.einsum("kjbc, kiac -> ijab", t, C)

    def get_Dijab(self, t):

        L = self.make_L()
        u = make_u(t)
        B = L + 0.5 * np.eisum("ilad, dclk -> icak", u, L)

        return 0.5 * np.einsum("jkbc, icak -> ijab", u, B)

    def get_Eijab(self, t):

        u = make_u(t)
        tmp_eri = np.einsum("Lld, Lkc -> ldkc", self.Lov, self.Lov)
        E = self.fvv - np.einsum("klbd, ldkc -> bc", u, tmp_eri)

        return np.einsum("ijac, bc -> ijab", t, E)

    def get_Gijab(self, t):

        u = make_u(t)
        tmp_eri = np.einsum("Lkd, Llc -> kdlc", self.Lov, self.Lov)
        G = self.foo + np.einsum("ljcd, kdlc -> jk", u, tmp_eri)

        return np.einsum("-ikab, jk -> ijab", t, G)

    def make_u(t):

        return 2 * t - np.transpose(t, (0, 1, 3, 2))

    def make_L(self):

        tmp_eri1 = np.einsum("Lpr ,Lqs -> rspq", self.fov, self.fov)
        tmp_eri2 = np.einsum("Lps ,Lqr -> rspq", self.fov, self.fov)

        return 2 * tmp_eri1 - tmp_eri2

    #####
