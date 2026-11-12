import numpy as np

from pyscf import df, lib


class DFRPA:
    """DFRPA model containing the occupied-virtual ladder contribution.

    The production ``active_jvv`` mode retains the complete ``Joo`` tensor
    while restricting both virtual indices of ``Jvv`` to an externally
    selected active-particle space.  The ``full`` mode is retained solely as
    an exact reference for validation.
    """

    def __init__(self, mf, eris, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self.ll_max_its = kwargs.get("ll_max_its", 50)
        self.ll_con_tol = kwargs.get("ll_con_tol", 1.0e-6)

        self.ladder_mode = kwargs.get("ladder_mode", "active_jvv").lower()
        if self.ladder_mode not in ("active_jvv", "full"):
            raise ValueError("ladder_mode must be 'active_jvv' or 'full'")

        self.act_hole = self._validate_active_indices(
            kwargs.get("act_hole"), eris.Loo.shape[1], "act_hole"
        )
        self.act_particle = self._validate_active_indices(
            kwargs.get("act_particle"), eris.Lvv.shape[1], "act_particle"
        )
        if self.ladder_mode == "active_jvv" and self.act_particle is None:
            raise ValueError(
                "ladder_mode='active_jvv' requires act_particle indices"
            )
        self._last_active_ladder_report = None

        self.diis = True
        self._eris = eris
        self._e_corr = None

    @property
    def e_tot(self):
        if self._e_corr is None:
            print("MPCC did not run, return mean field solution:")
            return float(self.mf.e_tot)
        return float(self.mf.e_tot + self._e_corr)

    @property
    def nvir(self):
        return self.mf.mol.nao - self.nocc

    @property
    def nocc(self):
        return self.mf.mol.nelec[0]

    @property
    def active_ladder_report(self):
        """Dimensions used by the most recent active-``Jvv`` contraction."""
        return self._last_active_ladder_report

    @staticmethod
    def _validate_active_indices(indices, size, name):
        if indices is None:
            return None
        indices = np.asarray(indices)
        if indices.dtype == bool:
            if indices.shape != (size,):
                raise ValueError(f"boolean {name} must have shape ({size},)")
            indices = np.flatnonzero(indices)
        else:
            if indices.ndim != 1 or not np.issubdtype(
                indices.dtype, np.integer
            ):
                raise ValueError(
                    f"{name} must be a one-dimensional index array"
                )
            indices = indices.astype(np.intp, copy=False)
        if indices.size == 0:
            raise ValueError(f"{name} must not be empty")
        if np.any(indices < 0) or np.any(indices >= size):
            raise ValueError(f"{name} contains an index outside [0, {size})")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} contains duplicate indices")
        return np.asarray(indices, dtype=np.intp)

    def kernel(self, t1=None, t2=None):
        if t1 is None and t2 is None:
            t1, t2 = self.init_amps()

        err = np.inf
        count = 0
        adiis = lib.diis.DIIS()
        adiis.space = 8

        while err > self.ll_con_tol and count < self.ll_max_its:
            res, t1_new, t2_new = self.updated_amps(t1, t2)
            if self.diis:
                t1_new, t2_new = self.run_diis(t1_new, t2_new, adiis)

            t1, t2 = t1_new, t2_new
            count += 1
            err = res
            print(f"It {count}; residual {res:.6e}")

        return t1, t2

    def updated_amps(self, t1, t2):
        X, Xoo, Xvo, Xvo_t2 = self.get_X(t1, t2)
        Joo, Jvo, Jvv = self.get_J(Xoo, Xvo, Xvo_t2, t1)
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)
        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Xvo_t2)

        omega = self.get_Ω(X, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2)
        res2 = self.update_t2(t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv)

        res1 = omega.T / self._eris.eia
        res2 = res2 / self._eris.D
        res = np.linalg.norm(res1) + np.linalg.norm(res2)

        t1 -= res1
        t2 -= res2
        return res, t1, t2

    def run_diis(self, t1, t2, adiis):
        vector = self.amplitudes_to_vector(t1, t2)
        return self.vector_to_amplitudes(adiis.update(vector))

    def amplitudes_to_vector(self, t1, t2, out=None):
        nov = self.nocc * self.nvir
        size = nov + nov * (nov + 1) // 2
        vector = np.ndarray(size, t1.dtype, buffer=out)
        vector[:nov] = t1.ravel()
        lib.pack_tril(
            t2.transpose(0, 2, 1, 3).reshape(nov, nov),
            out=vector[nov:],
        )
        return vector

    def vector_to_amplitudes(self, vector):
        nov = self.nocc * self.nvir
        t1 = vector[:nov].copy().reshape((self.nocc, self.nvir))
        t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
        t2 = t2.reshape(
            self.nocc, self.nvir, self.nocc, self.nvir
        ).transpose(0, 2, 1, 3)
        return t1, np.asarray(t2, order="C")

    def get_X(self, t1, t2):
        t2_antisym = 2.0 * t2 - t2.transpose(0, 1, 3, 2)

        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xvo_t2 = lib.einsum("Lkc,ikac->Lai", self._eris.Lov, t2_antisym)
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = 2.0 * lib.einsum("Lia,ia->L", self._eris.Lov, t1)
        return X, Xoo, Xvo, Xvo_t2

    def get_J(self, Xoo, Xvo, Xvo_t2, t1):
        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo
            + Xvo_t2
            + self._eris.Lvo
            - lib.einsum("Lji,ja->Lai", Joo, t1)
        )

        if self.ladder_mode == "active_jvv":
            active = self.act_particle
            Jvv = self._eris.Lvv[:, active][:, :, active].copy()
            Jvv -= lib.einsum(
                "Lkq,kp->Lpq",
                self._eris.Lov[:, :, active],
                t1[:, active],
            )
        else:
            Jvv = self._eris.Lvv - lib.einsum(
                "Lkb,ka->Lab", self._eris.Lov, t1
            )

        return Joo, Jvo, Jvv

    def get_F(self, t1, X, Xoo, Xvo):
        Foo = self._eris.foo.copy()
        Foo += lib.einsum("Lij,L->ij", self._eris.Loo, X)
        Foo -= lib.einsum("Lmj,Lim->ij", self._eris.Loo, Xoo)

        Fvv = self._eris.fvv.copy()
        Fvv += lib.einsum("Lab,L->ab", self._eris.Lvv, X)
        Fvv -= lib.einsum("Lmb,Lam->ab", self._eris.Lov, Xvo)

        Fov = self._eris.fov.copy()
        Fov += lib.einsum("Ljb,L->jb", self._eris.Lov, X)
        Fov -= lib.einsum("Lji,Lib->jb", Xoo, self._eris.Lov)
        return Foo, Fvv, Fov

    def add_t2_to_fock(self, Fvv, Foo, Xvo_t2):
        Foo += lib.einsum("Lie,Lej->ij", self._eris.Lov, Xvo_t2)
        Fvv -= lib.einsum("Lmb,Lam->ab", self._eris.Lov, Xvo_t2)
        return Foo, Fvv


    def get_Y2(self, t2, Joo, Jvv):

        tmp = lib.einsum("Lbc,ijac->Lijab", Jvv, t2)
        tmp -= lib.einsum("Lmi,mjab->Lijab", Joo, t2)

        Y2 += tmp + tmp.transpose(0, 2, 1, 4, 3)

        return Y2

    def get_Ω(self, X, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2):
        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy()
        Foo_tmp += 0.5 * lib.einsum("ic,jc->ij", Fov, t1)
        Fvv_tmp -= 0.5 * lib.einsum("lb,la->ab", Fov, t1)

        omega = self._eris.fov.T.copy()
        omega -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        omega -= lib.einsum("Laj,Lji->ai", Xvo_t2, self._eris.Loo)
        omega += lib.einsum("Lai,L->ai", self._eris.Lvo, X)
        omega += lib.einsum("Lae,Lei->ai", self._eris.Lvv, Xvo_t2)
        omega += lib.einsum("ib,ab->ai", t1, Fvv_tmp)
        omega -= lib.einsum("ka,ki->ai", t1, Foo_tmp)

        t2_antisym = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
        omega += lib.einsum("ijab,jb->ai", t2_antisym, Fov)
        return omega

    @staticmethod
    def ladder_contraction_full(t2, Joo, Jvv):
        """Evaluate the complete ladder contraction for reference."""
        interaction = lib.einsum("Lmj,Lbe->mbje", Joo, Jvv)
        return lib.einsum("mbje,imae->ijab", interaction, t2)

    @staticmethod
    def ladder_contraction_active_jvv(t2, Joo, Jvv, act_particle):
        """Evaluate the ladder using only the active-particle ``Jvv`` block.

        ``Joo`` remains complete.  Both indices of ``Jvv[L,b,e]`` are
        restricted to ``act_particle``.  The external ``a`` index of
        ``t2[i,m,a,e]`` remains in the full virtual space.
        """
        act_particle = np.asarray(act_particle, dtype=np.intp)
        nactive = act_particle.size
        if Jvv.shape[1:] == (t2.shape[3], t2.shape[3]):
            Jvv_active = Jvv[:, act_particle][:, :, act_particle]
        elif Jvv.shape[1:] == (nactive, nactive):
            Jvv_active = Jvv
        else:
            raise ValueError(
                "Jvv must contain either the full virtual block or the "
                "active-particle block"
            )

        interaction = lib.einsum("Lmj,Lpq->mjpq", Joo, Jvv_active)
        active_result = lib.einsum(
            "mjpq,imaq->ijap",
            interaction,
            t2[:, :, :, act_particle],
        )
        result = np.zeros_like(t2)
        result[:, :, :, act_particle] = active_result
        return result

    def ladder_contraction(self, t2, Joo, Jvv):
        if self.ladder_mode == "full":
            return self.ladder_contraction_full(t2, Joo, Jvv)

        result = self.ladder_contraction_active_jvv(
            t2, Joo, Jvv, self.act_particle
        )
        self._last_active_ladder_report = {
            "nocc": self.nocc,
            "nvir": self.nvir,
            "nact_hole": 0 if self.act_hole is None else self.act_hole.size,
            "nact_particle": self.act_particle.size,
            "act_hole": self.act_hole,
            "act_particle": self.act_particle,
        }
        return result

    def update_t2(self, t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv):
        res2 = lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy()
        Foo_tmp += lib.einsum("ic,jc->ij", Fov, t1)
        Fvv_tmp -= lib.einsum("lb,la->ab", Fov, t1)

        tmp = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("mi,mjab->ijab", Foo_tmp, t2)
        tmp -= self.ladder_contraction(t2, Joo, Jvv)
        res2 += tmp + tmp.transpose(1, 0, 3, 2)
        return res2

    def update_t3(self, t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv):
        res2 = lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy()
        Foo_tmp += lib.einsum("ic,jc->ij", Fov, t1)
        Fvv_tmp -= lib.einsum("lb,la->ab", Fov, t1)

        tmp = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("mi,mjab->ijab", Foo_tmp, t2)
        tmp -= self.ladder_contraction(t2, Joo, Jvv)
        res2 += tmp + tmp.transpose(1, 0, 3, 2)
        return res2


    def init_amps(self):
        t2 = lib.einsum("Lai,Lbj->ijab", self._eris.Lvo, self._eris.Lvo)
        t2 /= self._eris.D
        t1 = self._eris.fov / self._eris.eia
        return t1, t2

    def energy(self, t1, t2):
        """Return the correlation energy."""
        fock = self._eris.fov.copy()
        energy = 2.0 * lib.einsum("ia,ia", fock, t1)
        tau = lib.einsum("ia,jb->ijab", t1, t1)
        tau += t2
        eris_ovov = lib.einsum(
            "Lia,Ljb->iajb", self._eris.Lov, self._eris.Lov
        )
        energy += 2.0 * lib.einsum("ijab,iajb", tau, eris_ovov)
        energy -= lib.einsum("ijab,ibja", tau, eris_ovov)
        return energy.real
