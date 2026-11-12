import numpy as np
from pyscf import lib, df
from pyscf.ao2mo import _ao2mo
from pyscf.cc import ccsd

from pyscf.sylvester_cc import mpcc_tools


class _OrbitalSpace:
    """Minimal object accepted by the restricted CCSD frozen-MO helpers."""

    def __init__(self, mo_occ, frozen):
        self.mo_occ = mo_occ
        self.frozen = frozen
        self._nocc = None
        self._nmo = None


class ERIs:
    """Density-fitted MO integrals with CCSD-compatible frozen orbitals.

    ``frozen`` accepts ``None``, an integer number of lowest orbitals, or an
    explicit sequence of MO indices, matching :class:`pyscf.cc.ccsd.CCSD`.
    """

    def __init__(self, mf, mo_coeff=None, frozen=None, mo_occ=None):

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self.nao = mf.mol.nao
        self.naux = self.with_df.get_naoaux()
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        full_mo_coeff = np.asarray(mo_coeff)
        full_mo_occ = np.asarray(mo_occ)
        if full_mo_coeff.ndim != 2:
            raise ValueError("mo_coeff must be a two-dimensional array")
        if full_mo_coeff.shape[0] != self.nao:
            raise ValueError(
                "mo_coeff AO dimension does not match the molecule: "
                f"{full_mo_coeff.shape[0]} != {self.nao}"
            )
        if full_mo_occ.ndim != 1 or (
            full_mo_occ.size != full_mo_coeff.shape[1]
        ):
            raise ValueError(
                "mo_occ must be one-dimensional with one entry per MO"
            )
        if isinstance(frozen, (int, np.integer)) and frozen < 0:
            raise ValueError("frozen must be nonnegative")

        orbital_space = _OrbitalSpace(full_mo_occ, frozen)
        try:
            frozen_mask = ccsd.get_frozen_mask(orbital_space)
            nocc = ccsd.get_nocc(orbital_space)
            nmo = ccsd.get_nmo(orbital_space)
        except (AssertionError, IndexError) as error:
            raise ValueError(
                "invalid frozen orbital specification for the supplied MOs"
            ) from error

        active_mo_occ = full_mo_occ[frozen_mask]
        counted_nocc = int(np.count_nonzero(active_mo_occ > 0))
        if nmo != int(np.count_nonzero(frozen_mask)) or nocc != counted_nocc:
            raise ValueError("frozen orbital mask is inconsistent")
        if nocc < 1:
            raise ValueError("frozen specification removes all occupied MOs")
        if nmo <= nocc:
            raise ValueError("frozen specification removes all virtual MOs")
        if (
            np.any(active_mo_occ[:nocc] <= 0)
            or np.any(active_mo_occ[nocc:] > 0)
        ):
            raise ValueError(
                "restricted occupied and virtual MOs must be ordered"
            )

        self.frozen = frozen
        self.frozen_mask = np.asarray(frozen_mask, dtype=bool)
        self.active_orbital_indices = np.flatnonzero(self.frozen_mask)
        self.full_mo_coeff = full_mo_coeff
        self.full_mo_occ = full_mo_occ
        self.mo_coeff = np.asarray(
            full_mo_coeff[:, self.frozen_mask], order="F"
        )
        self.mo_occ = active_mo_occ
        self.nmo = nmo
        self.nocc = nocc
        self.nvir = nmo - nocc

        full_mo_energy = np.asarray(mf.mo_energy)
        if full_mo_energy.ndim != 1 or (
            full_mo_energy.size != full_mo_coeff.shape[1]
        ):
            raise ValueError(
                "mf.mo_energy must contain one value per supplied MO"
            )
        self.mo_energy = full_mo_energy[self.frozen_mask]

        fock_mo = self.mo_coeff.conj().T @ mf.get_fock() @ self.mo_coeff
        self.foo = fock_mo[: self.nocc, : self.nocc]
        self.fvv = fock_mo[self.nocc :, self.nocc :]
        self.fov = fock_mo[: self.nocc, self.nocc :]

        # NOTE change convetion to virtual, occupied 
        self.eia = lib.direct_sum(
            "-i+a->ia", np.diag(self.foo), np.diag(self.fvv)
        )

        # NOTE Shall we keep this call here?
        self._make_df_eris()
        self._make_df_denominator()

    def get_frozen_mask(self):
        """Return the CCSD-style mask of retained orbitals."""
        return self.frozen_mask.copy()

    def _make_df_denominator(self):

        self.dD = mpcc_tools.piv_chol_tensor(self.eia)
        
        # NOTE REMOVE THIS LATER!!!
        self.D = lib.direct_sum("ia+jb->ijab", self.eia, self.eia)

          
    def _make_df_eris(self):
        
        Loo = np.empty((self.naux, self.nocc, self.nocc))
        Lov = np.empty((self.naux, self.nocc, self.nvir))
        Lvv = np.empty((self.naux, self.nvir, self.nvir))
        mo = np.asarray(self.mo_coeff, order='F')
        ijslice = (0, self.nmo, 0, self.nmo)
        p1 = 0
        Lpq = None
        for k, eri1 in enumerate(self.with_df.loop()):
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Lpq = Lpq.reshape(p1 - p0, self.nmo, self.nmo)
            Loo[p0:p1] = Lpq[:, :self.nocc, :self.nocc]
            Lov[p0:p1] = Lpq[:, :self.nocc, self.nocc:]
            Lvv[p0:p1] = Lpq[:, self.nocc:, self.nocc:]
        Lpq = None
        Lvo = Lov.transpose(0,2,1).reshape(self.naux,self.nvir,self.nocc)
        self.Loo = Loo
        self.Lov = Lov
        self.Lvo = Lvo
        self.Lvv = Lvv       
