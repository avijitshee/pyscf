"""Factorized Laplace--Sylvester implementation of the DFRPA model.

The doubles amplitudes are represented during the iteration as

    t2[ijab] = -sum_R Y[Rai] Y[Rbj]

where the compound rank index ``R`` consists of the density-fitting and
Laplace-quadrature indices.  The persistent iterative state remains
factorized.  A dense doubles tensor is reconstructed transiently for the
faster direct and exchange contractions in ``Xvo_t2`` and released afterward.
The returned doubles tensor is reconstructed after the iteration.

Passing ``ladder_mode="perturbative"`` leaves the Sylvester iterations
ladder-free and adds the full ladder energy correction after convergence.  Its
ladder residual is assembled from the stored factors without applying the
dense ladder operator to ``t2``.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.special

from pyscf import lib
from pyscf.lib import logger

from pyscf.sylvester_cc import dfrpa, laplace_quadrature


class _IterationIntermediates(NamedTuple):
    """Arrays shared by the singles and doubles solves in one iteration."""

    X: np.ndarray
    Xoo: np.ndarray
    Xvo: np.ndarray
    Xvo_t2: np.ndarray
    Foo: np.ndarray
    Fvv: np.ndarray
    Fov: np.ndarray
    Fov_t1_oo: np.ndarray
    Fov_t1_vv: np.ndarray


class DFRPASylvester(dfrpa.DFRPA):
    """Canonical-RHF DFRPA with optional perturbative ladder energy."""

    def __init__(self, mf, eris, **kwargs):
        # The Sylvester model continues to use the historical ladder-free
        # DFRPA equations unless a ladder mode is explicitly requested.
        kwargs.setdefault("ladder_mode", None)
        super().__init__(mf, eris, **kwargs)
        if self.ladder_mode not in (None, "perturbative"):
            raise ValueError(
                "DFRPASylvester supports ladder_mode=None or "
                "ladder_mode='perturbative'"
            )

        self.verbose = getattr(mf, "verbose", logger.NOTE)
        self.stdout = getattr(mf, "stdout", None)
        self.diis = kwargs.get("diis", True)
        self.diis_space = int(kwargs.get("diis_space", 8))
        self.ll_laplace_quad_t1 = kwargs.get("ll_laplace_quad_t1")
        self.ll_laplace_quad_t2 = kwargs.get("ll_laplace_quad_t2")
        self.ll_laplace_root = kwargs.get("ll_laplace_root")
        self.ll_laplace_npoints = int(kwargs.get("ll_laplace_npoints", 16))
        self.ll_laplace_interval_margin = float(
            kwargs.get("ll_laplace_interval_margin", 0.1)
        )
        self.ll_chebyshev_tol = float(kwargs.get("ll_chebyshev_tol", 1.0e-12))
        self.ll_chebyshev_max_degree = int(
            kwargs.get("ll_chebyshev_max_degree", 300)
        )
        self.ll_canonical_tol = float(kwargs.get("ll_canonical_tol", 1.0e-8))

        if self.diis_space < 1:
            raise ValueError("diis_space must be positive")
        if self.ll_laplace_npoints < 1:
            raise ValueError("ll_laplace_npoints must be positive")
        if not 0.0 <= self.ll_laplace_interval_margin < 1.0:
            raise ValueError("ll_laplace_interval_margin must be in [0, 1)")

        self._quadrature_t1 = None
        self._quadrature_t2 = None
        self._Y = None
        self._e_tot = None
        self._xvo_t2_timing = None
        self._xvo_t2_timing_history = []
        self.converged = False

    @property
    def e_corr(self):
        return self._e_corr

    @property
    def xvo_t2_timing(self):
        """Timing breakdown for the most recent nonzero ``Xvo_t2`` build."""
        if self._xvo_t2_timing is None:
            return None
        return dict(self._xvo_t2_timing)

    def kernel(self, t1=None, t2=None):
        """Solve the coupled equations and return dense ``(t1, t2)``."""
        if t2 is not None:
            raise ValueError(
                "DFRPASylvester starts from factorized zero doubles and does "
                "not accept a dense t2 initial guess"
            )

        quad_t1, quad_t2 = self._get_fixed_quadratures()
        dtype = np.result_type(self._eris.Lvo, self._eris.fov)
        if t1 is None:
            t1 = np.zeros((self.nocc, self.nvir), dtype=dtype)
        else:
            t1 = np.asarray(t1, dtype=dtype).copy()
            if t1.shape != (self.nocc, self.nvir):
                raise ValueError(
                    f"t1 must have shape {(self.nocc, self.nvir)}, got {t1.shape}"
                )

        Y = np.zeros(
            (self._eris.Lvo.shape[0], quad_t2.nlap, self.nvir, self.nocc),
            dtype=dtype,
        )

        adiis = None
        if self.diis:
            adiis = lib.diis.DIIS()
            adiis.space = self.diis_space

        self.converged = False
        for iteration in range(1, self.ll_max_its + 1):
            # Use a simultaneous fixed-point update: both equations are
            # evaluated at the same current (t1, Y), so every common
            # intermediate, including Xvo_t2, is constructed only once.
            intermediates = self.get_iteration_intermediates(t1, Y)
            Y_target = self.solve_doubles(
                t1, Y, quad_t2, intermediates=intermediates
            )
            t1_target = self.solve_singles(
                t1, Y, quad_t1, intermediates=intermediates
            )

            delta_t1 = t1_target - t1
            delta_Y = Y_target - Y
            change_t1 = self._normalized_change(delta_t1)
            change_Y = self._normalized_change(delta_Y)

            logger.info(
                self,
                "DFRPASylvester iteration %d: |dt1|/sqrt(N) = %.6e; "
                "|dY|/sqrt(N) = %.6e",
                iteration,
                change_t1,
                change_Y,
            )

            if change_t1 < self.ll_con_tol and change_Y < self.ll_con_tol:
                t1, Y = t1_target, Y_target
                self.converged = True
                break

            vector = self.amplitudes_to_factor_vector(t1_target, Y_target)
            error = self.amplitudes_to_factor_vector(delta_t1, delta_Y)
            if adiis is not None:
                vector = adiis.update(vector, xerr=error)
            t1, Y = self.factor_vector_to_amplitudes(vector, Y_target.shape)

        if not self.converged:
            logger.warn(
                self,
                "DFRPASylvester did not converge in %d iterations; final "
                "normalized changes are dt1=%.6e and dY=%.6e",
                self.ll_max_its,
                change_t1,
                change_Y,
            )

        t2_dense = self.reconstruct_t2(Y)
        self._Y = Y
        self._e_corr = float(self.energy(t1, t2_dense))
        self._e_tot = float(self.mf.e_tot + self._e_corr)
        return t1, t2_dense

    @staticmethod
    def _normalized_change(change):
        if change.size == 0:
            return 0.0
        return float(np.linalg.norm(change) / np.sqrt(change.size))

    @staticmethod
    def amplitudes_to_factor_vector(t1, Y):
        return np.concatenate((np.ravel(t1), np.ravel(Y)))

    def factor_vector_to_amplitudes(self, vector, Y_shape):
        nt1 = self.nocc * self.nvir
        t1 = np.asarray(vector[:nt1]).reshape(self.nocc, self.nvir).copy()
        Y = np.asarray(vector[nt1:]).reshape(Y_shape).copy()
        return t1, Y

    def _get_fixed_quadratures(self):
        if self._quadrature_t1 is not None and self._quadrature_t2 is not None:
            return self._quadrature_t1, self._quadrature_t2

        gaps = np.asarray(self._eris.eia, dtype=float)
        if gaps.size == 0 or np.any(gaps <= 0.0):
            smallest = float(np.min(gaps)) if gaps.size else np.nan
            raise ValueError(
                "Laplace--Sylvester canonical orbital gaps must be positive; "
                f"smallest gap is {smallest:.6e}"
            )
        one_gap = (float(np.min(gaps)), float(np.max(gaps)))
        two_gap = (2.0 * one_gap[0], 2.0 * one_gap[1])

        self._quadrature_t1 = self._make_quadrature(
            self.ll_laplace_quad_t1, one_gap, "singles"
        )
        self._quadrature_t2 = self._make_quadrature(
            self.ll_laplace_quad_t2, two_gap, "doubles"
        )
        return self._quadrature_t1, self._quadrature_t2

    def _make_quadrature(self, configured, interval, equation):
        if configured is not None:
            quad = configured
        elif self.ll_laplace_root is not None:
            interval = self._generated_interval(interval)
            quad = laplace_quadrature.from_init_table(
                self.ll_laplace_root,
                interval[0],
                interval[1],
                self.ll_laplace_npoints,
            )
        else:
            bundled_root = (
                Path(__file__).resolve().parents[2] / "external" / "laplace-minimax"
            )
            if not (bundled_root / "data" / "init_para.txt").is_file():
                raise ValueError(
                    "No Laplace quadrature configured. Provide both "
                    "ll_laplace_quad_t1 and ll_laplace_quad_t2, or provide "
                    "ll_laplace_root."
                )
            interval = self._generated_interval(interval)
            quad = laplace_quadrature.from_init_table(
                bundled_root,
                interval[0],
                interval[1],
                self.ll_laplace_npoints,
            )

        self._validate_quadrature(quad, interval, equation)
        return quad

    def _generated_interval(self, interval):
        margin = self.ll_laplace_interval_margin
        return ((1.0 - margin) * interval[0], (1.0 + margin) * interval[1])

    @staticmethod
    def _validate_quadrature(quad, interval, equation):
        required = ("exponents", "weights", "ymin", "ymax", "nlap")
        if any(not hasattr(quad, name) for name in required):
            raise TypeError(
                f"{equation} Laplace quadrature must be a LaplaceQuadrature-like "
                "object"
            )
        if np.any(np.asarray(quad.weights) < 0.0):
            raise ValueError(
                f"{equation} factorized Laplace quadrature requires "
                "nonnegative weights"
            )
        if np.any(np.asarray(quad.exponents) < 0.0):
            raise ValueError(
                f"{equation} Laplace quadrature requires nonnegative exponents"
            )
        tol = 100.0 * np.finfo(float).eps * max(1.0, float(quad.ymax))
        if interval[0] < quad.ymin - tol or interval[1] > quad.ymax + tol:
            raise ValueError(
                f"{equation} Laplace quadrature interval "
                f"[{quad.ymin:.6e}, {quad.ymax:.6e}] does not cover required "
                f"gaps [{interval[0]:.6e}, {interval[1]:.6e}]"
            )

    def get_X_t1(self, t1):
        """Build the X intermediates that are independent of doubles."""
        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = 2.0 * lib.einsum("Lia,ia->L", self._eris.Lov, t1)
        return X, Xoo, Xvo

    def get_iteration_intermediates(self, t1, Y):
        """Build all arrays shared by the two simultaneous amplitude solves."""
        X, Xoo, Xvo = self.get_X_t1(t1)
        Xvo_t2 = self.get_Xvo_t2(Y)
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)
        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Xvo_t2)
        Fov_t1_oo = lib.einsum("ic,jc->ij", Fov, t1)
        Fov_t1_vv = lib.einsum("lb,la->ab", Fov, t1)
        return _IterationIntermediates(
            X,
            Xoo,
            Xvo,
            Xvo_t2,
            Foo,
            Fvv,
            Fov,
            Fov_t1_oo,
            Fov_t1_vv,
        )

    def get_X_factorized(self, t1, Y):
        X, Xoo, Xvo = self.get_X_t1(t1)
        Xvo_t2 = self.get_Xvo_t2(Y)
        return X, Xoo, Xvo, Xvo_t2

    def get_Xvo_t2(self, Y):
        """Evaluate ``Lov * (2*t2 - t2.swapaxes(2,3))`` through dense ``t2``."""
        Y_flat = self._flatten_factors(Y)
        if not np.any(Y_flat):
            return np.zeros_like(self._eris.Lvo)
        return self.get_Xvo_t2_dense(Y_flat)

    def get_Xvo_t2_dense(self, Y):
        """Reconstruct transient ``t2`` and time its direct/exchange actions."""
        Y_flat = self._flatten_factors(Y)
        total_start = logger.perf_counter()

        reconstruction_start = logger.perf_counter()
        t2 = self.reconstruct_t2(Y_flat)
        reconstruction_time = logger.perf_counter() - reconstruction_start

        direct_start = logger.perf_counter()
        Xvo_t2 = 2.0 * lib.einsum("Lkc,ikac->Lai", self._eris.Lov, t2)
        direct_time = logger.perf_counter() - direct_start

        exchange_start = logger.perf_counter()
        exchange = lib.einsum("Lkc,ikca->Lai", self._eris.Lov, t2)
        Xvo_t2 -= exchange
        exchange_time = logger.perf_counter() - exchange_start

        timing = {
            "direct": direct_time,
            "reconstruct_t2": reconstruction_time,
            "exchange": exchange_time,
            "total": logger.perf_counter() - total_start,
            "dense_t2_bytes": t2.nbytes,
        }
        self._xvo_t2_timing = timing
        if not hasattr(self, "_xvo_t2_timing_history"):
            self._xvo_t2_timing_history = []
        self._xvo_t2_timing_history.append(timing)
        logger.debug(
            self,
            "DFRPASylvester Xvo_t2 wall time: direct %.6f s; reconstruct "
            "%.6f s; exchange %.6f s; total %.6f s",
            timing["direct"],
            timing["reconstruct_t2"],
            timing["exchange"],
            timing["total"],
        )
        return Xvo_t2

    def get_J_factorized(self, Xoo, Xvo, Xvo_t2, t1):
        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo
            + Xvo_t2
            + self._eris.Lvo
            - lib.einsum("Lji,ja->Lai", Joo, t1)
        )
        return Joo, Jvo

    def ladder_pert_energy(self, t2, Joo, Jvv):
        """Evaluate the post-convergence ladder correction from ``Y``.

        The contraction order is

        ``Joo,Jvv -> ovov; ovov,Y -> ovLR; ovLR,Y -> ladder_res``.

        The final scalar contraction uses the supplied dense ``t2``, matching
        :meth:`dfrpa.DFRPA.ladder_pert_energy` without constructing the dense
        ladder action ``ovov * t2``.
        """
        if self._Y is None:
            raise RuntimeError(
                "factorized perturbative ladder energy requires converged Y "
                "factors; call kernel() first"
            )

        Y = np.asarray(self._Y)
        if Y.ndim != 4:
            raise ValueError(
                "stored Y factors must have shape "
                "(naux,nlap,nvir,nocc)"
            )

        ovov = lib.einsum("Qmj,Qbe->mbje", Joo, Jvv)
        ovLR = lib.einsum("mbje,LRem->bjLR", ovov, Y)
        del ovov
        tmp = lib.einsum("bjLR,LRai->ijab", ovLR, Y)
        del ovLR
        ladder_res = tmp + tmp.transpose(1, 0, 3, 2)

        energy = 2.0 * lib.einsum("ijab,ijab", t2, ladder_res)
        energy -= lib.einsum("ijab,ijba", t2, ladder_res)
        return energy.real

    def solve_doubles(self, t1, Y, quad, intermediates=None):
        """Solve doubles, optionally reusing all iteration intermediates."""
        if intermediates is None:
            intermediates = self.get_iteration_intermediates(t1, Y)
        _, Jvo = self.get_J_factorized(
            intermediates.Xoo,
            intermediates.Xvo,
            intermediates.Xvo_t2,
            t1,
        )

        Foo_dressed = intermediates.Foo + intermediates.Fov_t1_oo
        Fvv_dressed = intermediates.Fvv - intermediates.Fov_t1_vv
        self._validate_dressed_gaps(Foo_dressed, Fvv_dressed, quad, 2, "doubles")
        return self.get_sylvester_laplace_matrix_factors(
            Jvo, Foo_dressed, Fvv_dressed, quad
        )

    def get_omega0_factorized(self, X, Xvo, Xvo_t2, Fov, Y):
        """Build the singles source with both explicit T1 actions removed."""
        omega0 = self._eris.fov.T.copy()
        omega0 -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        omega0 -= lib.einsum("Laj,Lji->ai", Xvo_t2, self._eris.Loo)
        omega0 += lib.einsum("Lai,L->ai", self._eris.Lvo, X)
        omega0 += lib.einsum("Lae,Lei->ai", self._eris.Lvv, Xvo_t2)
        omega0 += self.get_t2_fov(Y, Fov)
        return omega0

    @staticmethod
    def get_t2_fov(Y, Fov):
        """Evaluate ``(2*t2[ijab]-t2[ijba]) Fov[jb]`` from factors."""
        Y_flat = DFRPASylvester._flatten_factors(Y)
        direct = lib.einsum("Rbj,jb->R", Y_flat, Fov)
        result = -2.0 * lib.einsum("R,Rai->ai", direct, Y_flat)
        exchange = lib.einsum("Rbi,jb->Rij", Y_flat, Fov)
        result += lib.einsum("Rij,Raj->ai", exchange, Y_flat)
        return result

    def solve_singles(self, t1, Y, quad, intermediates=None):
        """Solve singles, optionally reusing all iteration intermediates."""
        if intermediates is None:
            intermediates = self.get_iteration_intermediates(t1, Y)

        Foo_dressed = intermediates.Foo + 0.5 * intermediates.Fov_t1_oo
        Fvv_dressed = intermediates.Fvv - 0.5 * intermediates.Fov_t1_vv
        omega0 = self.get_omega0_factorized(
            intermediates.X,
            intermediates.Xvo,
            intermediates.Xvo_t2,
            intermediates.Fov,
            Y,
        )
        self._validate_dressed_gaps(Foo_dressed, Fvv_dressed, quad, 1, "singles")
        return self.solve_singles_laplace(
            omega0, Foo_dressed, Fvv_dressed, quad
        ).T

    def solve_singles_laplace(self, omega0, Foo, Fvv, quad):
        """Return ``T1.T`` from the factorized Laplace Sylvester inverse."""
        exp_v = self._chebyshev_exponentials(
            Fvv, -np.asarray(quad.exponents), self.ll_chebyshev_tol,
            self.ll_chebyshev_max_degree,
        )
        exp_o = self._chebyshev_exponentials(
            Foo, np.asarray(quad.exponents), self.ll_chebyshev_tol,
            self.ll_chebyshev_max_degree,
        )
        result = np.zeros_like(omega0)
        for weight, left, right in zip(quad.weights, exp_v, exp_o):
            result -= weight * (left @ omega0 @ right)
        return result

    def get_sylvester_laplace_matrix_factors(self, Jvo, Foo, Fvv, quad):
        """Apply actual (possibly nonsymmetric) dressed Fock exponentials."""
        exp_v = self._chebyshev_exponentials(
            Fvv, -np.asarray(quad.exponents), self.ll_chebyshev_tol,
            self.ll_chebyshev_max_degree,
        )
        exp_o = self._chebyshev_exponentials(
            Foo, np.asarray(quad.exponents), self.ll_chebyshev_tol,
            self.ll_chebyshev_max_degree,
        )

        factors = np.empty(
            (Jvo.shape[0], quad.nlap, Jvo.shape[1], Jvo.shape[2]),
            dtype=np.result_type(Jvo, Foo, Fvv),
        )
        for index, (weight, left, right) in enumerate(
            zip(quad.weights, exp_v, exp_o)
        ):
            factors[:, index] = np.sqrt(weight) * lib.einsum(
                "ab,Lbi,ij->Laj", left, Jvo, right
            )
        return factors

    @classmethod
    def _chebyshev_exponentials(cls, matrix, scales, tol, max_degree):
        matrix = np.asarray(matrix)
        lower, upper = cls._symmetric_spectral_bounds(matrix)
        center = 0.5 * (upper + lower)
        radius = 0.5 * (upper - lower)
        if radius <= 0.0:
            radius = np.finfo(float).eps * max(1.0, abs(center))

        coefficients = [
            cls._chebyshev_exp_coefficients(
                scale, center, radius, tol, max_degree
            )
            for scale in scales
        ]
        max_terms = max(len(value) for value in coefficients)
        identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
        scaled = (matrix - center * identity) / radius
        sequence = [identity]
        if max_terms > 1:
            sequence.append(scaled)
        for _ in range(2, max_terms):
            sequence.append(2.0 * scaled @ sequence[-1] - sequence[-2])
        sequence = np.asarray(sequence)
        return [
            np.tensordot(value, sequence[: len(value)], axes=(0, 0))
            for value in coefficients
        ]

    @staticmethod
    def _symmetric_spectral_bounds(matrix):
        """Bounds for scaling only; the matrix action itself is not symmetrized."""
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("dressed Fock matrix must be square")
        values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        lower = float(values[0])
        upper = float(values[-1])
        pad = 100.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
        if upper - lower < 2.0 * pad:
            lower -= pad
            upper += pad
        return lower, upper

    @staticmethod
    def _chebyshev_exp_coefficients(scale, center, radius, tol, max_degree):
        prefactor = np.exp(scale * center)
        beta = scale * radius
        coefficients = [prefactor * scipy.special.iv(0, beta)]
        for degree in range(1, max_degree + 1):
            coefficient = 2.0 * prefactor * scipy.special.iv(degree, beta)
            coefficients.append(coefficient)
            if abs(coefficient) <= tol * max(1.0, abs(coefficients[0])):
                return np.asarray(coefficients)
        raise RuntimeError(
            "Chebyshev exponential action did not converge within "
            f"max_degree={max_degree}"
        )

    def _validate_dressed_gaps(
        self, Foo, Fvv, quad, multiplier, equation
    ):
        lower_v, upper_v = self._symmetric_spectral_bounds(Fvv)
        lower_o, upper_o = self._symmetric_spectral_bounds(Foo)
        interval = (
            multiplier * (lower_v - upper_o),
            multiplier * (upper_v - lower_o),
        )
        if interval[0] <= 0.0:
            raise ValueError(
                f"{equation} dressed orbital gaps must be positive; smallest "
                f"estimated gap is {interval[0]:.6e}"
            )
        self._validate_quadrature(quad, interval, f"{equation} dressed")
        return interval

    @staticmethod
    def _flatten_factors(Y):
        Y = np.asarray(Y)
        if Y.ndim == 4:
            return Y.reshape(-1, Y.shape[-2], Y.shape[-1])
        if Y.ndim == 3:
            return Y
        raise ValueError("Y must have shape (naux,nlap,nvir,nocc) or (rank,nvir,nocc)")

    @staticmethod
    def reconstruct_t2(Y):
        Y_flat = DFRPASylvester._flatten_factors(Y)
        return -lib.einsum("Rai,Rbj->ijab", Y_flat, Y_flat)
