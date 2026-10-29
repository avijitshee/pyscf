"""Factorized Laplace--Sylvester implementation of the DFRPA model.

The doubles amplitudes are represented during the iteration as

    t2[ijab] = -sum_R Y[Rai] Y[Rbj]

where the compound rank index ``R`` consists of the density-fitting and
Laplace-quadrature indices.  The persistent iterative state remains
factorized.  By default, a dense doubles tensor is reconstructed transiently
for the otherwise expensive exchange part of ``Xvo_t2`` and released after
that contraction; the returned doubles tensor is reconstructed after the
iteration.
"""

from pathlib import Path

import numpy as np
import scipy.special

from pyscf import lib
from pyscf.lib import logger

from pyscf.sylvester_cc import dfrpa, laplace_quadrature


class DFRPASylvester(dfrpa.DFRPA):
    """Canonical-RHF DFRPA solved with factorized Sylvester equations."""

    def __init__(self, mf, eris, **kwargs):
        super().__init__(mf, eris, **kwargs)

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
        self.ll_xvo_t2_mode = kwargs.get("ll_xvo_t2_mode", "dense").lower()
        exchange_tol = kwargs.get("ll_exchange_screen_tol", 0.0)
        self.ll_exchange_screen_tol = 0.0 if exchange_tol is None else float(exchange_tol)
        self.ll_exchange_screen_mode = kwargs.get(
            "ll_exchange_screen_mode", "calibrated"
        ).lower()
        self.ll_exchange_q_block_size = int(
            kwargs.get("ll_exchange_q_block_size", 128)
        )
        self.ll_exchange_rank_block_size = int(
            kwargs.get("ll_exchange_rank_block_size", 512)
        )
        self.ll_exchange_check_interval = int(
            kwargs.get("ll_exchange_check_interval", 4)
        )
        self.ll_exchange_screen_refresh = int(
            kwargs.get("ll_exchange_screen_refresh", 4)
        )

        if self.diis_space < 1:
            raise ValueError("diis_space must be positive")
        if self.ll_laplace_npoints < 1:
            raise ValueError("ll_laplace_npoints must be positive")
        if not 0.0 <= self.ll_laplace_interval_margin < 1.0:
            raise ValueError("ll_laplace_interval_margin must be in [0, 1)")
        if self.ll_exchange_screen_tol < 0.0:
            raise ValueError("ll_exchange_screen_tol must be nonnegative")
        if self.ll_xvo_t2_mode not in ("dense", "factorized"):
            raise ValueError("ll_xvo_t2_mode must be 'dense' or 'factorized'")
        if self.ll_exchange_screen_mode not in ("bound", "calibrated"):
            raise ValueError(
                "ll_exchange_screen_mode must be 'bound' or 'calibrated'"
            )
        if self.ll_exchange_q_block_size < 1:
            raise ValueError("ll_exchange_q_block_size must be positive")
        if self.ll_exchange_rank_block_size < 1:
            raise ValueError("ll_exchange_rank_block_size must be positive")
        if self.ll_exchange_check_interval < 1:
            raise ValueError("ll_exchange_check_interval must be positive")
        if self.ll_exchange_screen_refresh < 0:
            raise ValueError("ll_exchange_screen_refresh must be nonnegative")

        self._validate_canonical_orbitals()
        self._quadrature_t1 = None
        self._quadrature_t2 = None
        self._Y = None
        self._e_tot = None
        self._exchange_screen_history = []
        self._exchange_screen_selection = None
        self._exchange_screen_calls = 0
        self.converged = False

    @property
    def e_corr(self):
        return self._e_corr

    def _validate_canonical_orbitals(self):
        """Reject occupied/virtual rotations of the canonical RHF orbitals."""
        foo_offdiag = self._eris.foo - np.diag(np.diag(self._eris.foo))
        fvv_offdiag = self._eris.fvv - np.diag(np.diag(self._eris.fvv))
        scale = max(
            1.0,
            float(np.linalg.norm(self._eris.foo)),
            float(np.linalg.norm(self._eris.fvv)),
        )
        error = max(
            float(np.linalg.norm(foo_offdiag)),
            float(np.linalg.norm(fvv_offdiag)),
            float(np.linalg.norm(self._eris.fov)),
        )
        if error > self.ll_canonical_tol * scale:
            raise ValueError(
                "DFRPASylvester requires canonical RHF orbitals; the MO Fock "
                f"matrix is not diagonal (relative off-diagonal norm "
                f"{error / scale:.3e})."
            )

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
            Y_target = self.solve_doubles(t1, Y, quad_t2)
            t1_target = self.solve_singles(t1, Y_target, quad_t1)

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

    def get_X_factorized(self, t1, Y):
        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xvo_t2 = self.get_Xvo_t2(Y)
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = 2.0 * lib.einsum("Lia,ia->L", self._eris.Lov, t1)
        return X, Xoo, Xvo, Xvo_t2

    def get_Xvo_t2(self, Y):
        """Evaluate ``Lov * (2*t2 - t2.swapaxes(2,3))`` from factors."""
        Y_flat = self._flatten_factors(Y)
        if not np.any(Y_flat):
            result = np.zeros_like(self._eris.Lvo)
            self._record_exchange_screen_stats(
                {
                    "zero_factors": True,
                    "total_blocks": 0,
                    "evaluated_blocks": 0,
                    "skipped_blocks": 0,
                    "screen_tolerance": self.ll_exchange_screen_tol,
                    "converged": True,
                    "exchange_norm": 0.0,
                    "remaining_bound": 0.0,
                    "relative_error_bound": 0.0,
                    "convergence": (),
                }
            )
            return result

        if self.ll_xvo_t2_mode == "dense":
            return self.get_Xvo_t2_dense(Y_flat)

        overlap = lib.einsum("Lkc,Rck->LR", self._eris.Lov, Y_flat)
        Xvo_t2 = -2.0 * lib.einsum("LR,Rai->Lai", overlap, Y_flat)
        Xvo_t2 += self.get_Xvo_t2_exchange(Y_flat)
        return Xvo_t2

    def get_Xvo_t2_dense(self, Y):
        """Reconstruct transient ``t2`` and contract ``Xvo_t2`` densely."""
        t2 = self.reconstruct_t2(Y)
        Xvo_t2 = 2.0 * lib.einsum("Lkc,ikac->Lai", self._eris.Lov, t2)
        exchange = lib.einsum("Lkc,ikca->Lai", self._eris.Lov, t2)
        Xvo_t2 -= exchange
        self._record_exchange_screen_stats(
            {
                "mode": "dense",
                "zero_factors": False,
                "total_blocks": 1,
                "evaluated_blocks": 1,
                "skipped_blocks": 0,
                "screen_tolerance": 0.0,
                "converged": True,
                "exchange_norm": float(np.linalg.norm(exchange)),
                "remaining_bound": 0.0,
                "relative_error_bound": 0.0,
                "dense_t2_bytes": t2.nbytes,
                "convergence": (),
            }
        )
        return Xvo_t2

    def get_Xvo_t2_exchange(self, Y):
        """Build the exchange part of ``Xvo_t2`` in dominant-first blocks.

        Each candidate block has the rigorous Frobenius upper bound

        ``||L_Q||_F * sum_R ||Y_R||_F**2``.

        Blocks are evaluated from largest to smallest bound.  If
        ``ll_exchange_screen_tol`` is positive, the evaluation stops when the
        bound on all omitted blocks guarantees the requested relative error
        in the accumulated exchange tensor.  A zero tolerance evaluates all
        blocks and is therefore exact up to contraction ordering.
        """
        Y_flat = self._flatten_factors(Y)
        Lov = self._eris.Lov
        naux = Lov.shape[0]
        nrank = Y_flat.shape[0]
        qsize = self.ll_exchange_q_block_size
        rsize = self.ll_exchange_rank_block_size

        q_ranges = [
            (start, min(start + qsize, naux))
            for start in range(0, naux, qsize)
        ]
        r_ranges = [
            (start, min(start + rsize, nrank))
            for start in range(0, nrank, rsize)
        ]
        if (
            self.ll_exchange_screen_tol > 0.0
            and self.ll_exchange_screen_mode == "calibrated"
        ):
            return self._get_Xvo_t2_exchange_calibrated(
                Y_flat, q_ranges, r_ranges
            )

        lnorm = np.asarray(
            [np.linalg.norm(Lov[q0:q1]) for q0, q1 in q_ranges]
        )
        ynorm2 = np.asarray(
            [lib.einsum("Rai,Rai->", Y_flat[r0:r1], Y_flat[r0:r1])
             for r0, r1 in r_ranges]
        )

        candidates = [
            (float(lnorm[qindex] * ynorm2[rindex]), qindex, rindex)
            for qindex in range(len(q_ranges))
            for rindex in range(len(r_ranges))
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)

        remaining_by_q = lnorm * float(np.sum(ynorm2))
        exchange = np.zeros(
            (naux, Y_flat.shape[1], Y_flat.shape[2]),
            dtype=np.result_type(Lov, Y_flat),
        )
        tolerance = self.ll_exchange_screen_tol
        convergence = []
        evaluated = 0
        converged = tolerance == 0.0 and not candidates

        for bound, qindex, rindex in candidates:
            q0, q1 = q_ranges[qindex]
            r0, r1 = r_ranges[rindex]
            Y_block = Y_flat[r0:r1]
            intermediate = lib.einsum(
                "Qkc,Rci->QRki", Lov[q0:q1], Y_block
            )
            exchange[q0:q1] += lib.einsum(
                "QRki,Rak->Qai", intermediate, Y_block
            )
            evaluated += 1
            remaining_by_q[qindex] = max(
                0.0, remaining_by_q[qindex] - bound
            )

            should_check = (
                tolerance > 0.0
                and (
                    evaluated % self.ll_exchange_check_interval == 0
                    or evaluated == len(candidates)
                )
            )
            if should_check:
                exchange_norm = float(np.linalg.norm(exchange))
                remaining_bound = float(np.linalg.norm(remaining_by_q))
                relative_bound = self._relative_remainder_bound(
                    exchange_norm, remaining_bound
                )
                convergence.append(
                    {
                        "evaluated_blocks": evaluated,
                        "exchange_norm": exchange_norm,
                        "remaining_bound": remaining_bound,
                        "relative_error_bound": relative_bound,
                    }
                )
                if relative_bound <= tolerance:
                    converged = True
                    break

        exchange_norm = float(np.linalg.norm(exchange))
        remaining_bound = float(np.linalg.norm(remaining_by_q))
        relative_bound = self._relative_remainder_bound(
            exchange_norm, remaining_bound
        )
        if tolerance == 0.0:
            converged = evaluated == len(candidates)

        stats = {
            "mode": "bound",
            "zero_factors": False,
            "total_blocks": len(candidates),
            "evaluated_blocks": evaluated,
            "skipped_blocks": len(candidates) - evaluated,
            "screen_tolerance": tolerance,
            "converged": converged,
            "exchange_norm": exchange_norm,
            "remaining_bound": remaining_bound,
            "relative_error_bound": relative_bound,
            "largest_block_bound": candidates[0][0] if candidates else 0.0,
            "smallest_evaluated_bound": (
                candidates[evaluated - 1][0] if evaluated else 0.0
            ),
            "q_block_size": qsize,
            "rank_block_size": rsize,
            "convergence": tuple(convergence),
        }
        self._record_exchange_screen_stats(stats)
        logger.debug(
            self,
            "DFRPASylvester exchange screening: evaluated %d/%d blocks; "
            "relative error bound %.3e (target %.3e)",
            evaluated,
            len(candidates),
            relative_bound,
            tolerance,
        )
        return exchange

    def _get_Xvo_t2_exchange_calibrated(self, Y_flat, q_ranges, r_ranges):
        """Select dominant exchange blocks from a measured convergence curve.

        A calibration build evaluates every block, sorts the actual block
        norms, and retains the shortest checked prefix whose exchange tensor
        differs from the complete build by no more than the configured
        relative tolerance.  The selected list is reused between calibration
        builds.  This mode is empirical between refreshes; use ``mode='bound'``
        when a rigorous per-call omitted-block bound is required.
        """
        self._exchange_screen_calls += 1
        refresh = self.ll_exchange_screen_refresh
        selection_key = (
            self._eris.Lov.shape,
            Y_flat.shape,
            self.ll_exchange_q_block_size,
            self.ll_exchange_rank_block_size,
            self.ll_exchange_screen_tol,
        )
        cached = self._exchange_screen_selection
        needs_calibration = (
            cached is None
            or cached["key"] != selection_key
            or (
                refresh > 0
                and self._exchange_screen_calls - cached["call"] >= refresh
            )
        )

        if not needs_calibration:
            exchange = np.zeros(
                (self._eris.Lov.shape[0], Y_flat.shape[1], Y_flat.shape[2]),
                dtype=np.result_type(self._eris.Lov, Y_flat),
            )
            for qindex, rindex in cached["blocks"]:
                self._accumulate_exchange_block(
                    exchange, Y_flat, q_ranges, r_ranges, qindex, rindex
                )
            stats = {
                "mode": "calibrated",
                "calibration_build": False,
                "zero_factors": False,
                "total_blocks": len(q_ranges) * len(r_ranges),
                "evaluated_blocks": len(cached["blocks"]),
                "retained_blocks": len(cached["blocks"]),
                "skipped_blocks": (
                    len(q_ranges) * len(r_ranges) - len(cached["blocks"])
                ),
                "screen_tolerance": self.ll_exchange_screen_tol,
                "converged": True,
                "exchange_norm": float(np.linalg.norm(exchange)),
                "remaining_bound": np.nan,
                "relative_error_bound": np.nan,
                "calibration_relative_error": cached["relative_error"],
                "q_block_size": self.ll_exchange_q_block_size,
                "rank_block_size": self.ll_exchange_rank_block_size,
                "convergence": cached["convergence"],
            }
            self._record_exchange_screen_stats(stats)
            return exchange

        full = np.zeros(
            (self._eris.Lov.shape[0], Y_flat.shape[1], Y_flat.shape[2]),
            dtype=np.result_type(self._eris.Lov, Y_flat),
        )
        contributions = []
        for qindex in range(len(q_ranges)):
            q0, q1 = q_ranges[qindex]
            for rindex in range(len(r_ranges)):
                contribution = self._exchange_block(
                    Y_flat, q_ranges, r_ranges, qindex, rindex
                )
                full[q0:q1] += contribution
                contributions.append(
                    (
                        float(np.linalg.norm(contribution)),
                        qindex,
                        rindex,
                        contribution,
                    )
                )

        contributions.sort(key=lambda item: item[0], reverse=True)
        full_norm = float(np.linalg.norm(full))
        partial = np.zeros_like(full)
        selected = []
        convergence = []
        relative_error = np.inf
        for index, (_, qindex, rindex, contribution) in enumerate(
            contributions, 1
        ):
            q0, q1 = q_ranges[qindex]
            partial[q0:q1] += contribution
            selected.append((qindex, rindex))
            should_check = (
                index % self.ll_exchange_check_interval == 0
                or index == len(contributions)
            )
            if should_check:
                omitted_norm = float(np.linalg.norm(full - partial))
                if full_norm == 0.0:
                    relative_error = 0.0 if omitted_norm == 0.0 else np.inf
                else:
                    relative_error = omitted_norm / full_norm
                convergence.append(
                    {
                        "evaluated_blocks": index,
                        "exchange_norm": float(np.linalg.norm(partial)),
                        "relative_error": relative_error,
                    }
                )
                if relative_error <= self.ll_exchange_screen_tol:
                    break

        cached = {
            "key": selection_key,
            "call": self._exchange_screen_calls,
            "blocks": tuple(selected),
            "relative_error": relative_error,
            "convergence": tuple(convergence),
        }
        self._exchange_screen_selection = cached
        total_blocks = len(contributions)
        stats = {
            "mode": "calibrated",
            "calibration_build": True,
            "zero_factors": False,
            "total_blocks": total_blocks,
            # Calibration computes every block once even if fewer are retained.
            "evaluated_blocks": total_blocks,
            "retained_blocks": len(selected),
            "skipped_blocks": total_blocks - len(selected),
            "screen_tolerance": self.ll_exchange_screen_tol,
            "converged": relative_error <= self.ll_exchange_screen_tol,
            "exchange_norm": float(np.linalg.norm(partial)),
            "complete_exchange_norm": full_norm,
            "remaining_bound": np.nan,
            "relative_error_bound": np.nan,
            "calibration_relative_error": relative_error,
            "largest_block_norm": contributions[0][0] if contributions else 0.0,
            "smallest_retained_block_norm": (
                contributions[len(selected) - 1][0] if selected else 0.0
            ),
            "q_block_size": self.ll_exchange_q_block_size,
            "rank_block_size": self.ll_exchange_rank_block_size,
            "convergence": tuple(convergence),
        }
        self._record_exchange_screen_stats(stats)
        logger.debug(
            self,
            "DFRPASylvester calibrated exchange screening: retained %d/%d "
            "blocks; measured relative error %.3e (target %.3e)",
            len(selected),
            total_blocks,
            relative_error,
            self.ll_exchange_screen_tol,
        )
        return partial

    def _exchange_block(self, Y_flat, q_ranges, r_ranges, qindex, rindex):
        q0, q1 = q_ranges[qindex]
        r0, r1 = r_ranges[rindex]
        Y_block = Y_flat[r0:r1]
        intermediate = lib.einsum(
            "Qkc,Rci->QRki", self._eris.Lov[q0:q1], Y_block
        )
        return lib.einsum("QRki,Rak->Qai", intermediate, Y_block)

    def _accumulate_exchange_block(
        self, exchange, Y_flat, q_ranges, r_ranges, qindex, rindex
    ):
        q0, q1 = q_ranges[qindex]
        exchange[q0:q1] += self._exchange_block(
            Y_flat, q_ranges, r_ranges, qindex, rindex
        )

    @staticmethod
    def _relative_remainder_bound(partial_norm, remaining_bound):
        """Bound ``||omitted|| / ||complete||`` using a partial sum."""
        complete_lower_bound = partial_norm - remaining_bound
        if remaining_bound == 0.0:
            return 0.0
        if complete_lower_bound <= 0.0:
            return np.inf
        return remaining_bound / complete_lower_bound

    def _record_exchange_screen_stats(self, stats):
        if not hasattr(self, "_exchange_screen_history"):
            self._exchange_screen_history = []
        self._exchange_screen_history.append(stats)
        self._exchange_screen_stats = stats

    def get_J_factorized(self, Xoo, Xvo, Xvo_t2, t1):
        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo
            + Xvo_t2
            + self._eris.Lvo
            - lib.einsum("Lji,ja->Lai", Joo, t1)
        )
        return Joo, Jvo

    def solve_doubles(self, t1, Y, quad):
        X, Xoo, Xvo, Xvo_t2 = self.get_X_factorized(t1, Y)
        _, Jvo = self.get_J_factorized(Xoo, Xvo, Xvo_t2, t1)
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)
        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Xvo_t2)

        Foo_dressed = Foo + lib.einsum("ic,jc->ij", Fov, t1)
        Fvv_dressed = Fvv - lib.einsum("lb,la->ab", Fov, t1)
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

    def solve_singles(self, t1, Y, quad):
        X, Xoo, Xvo, Xvo_t2 = self.get_X_factorized(t1, Y)
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)
        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Xvo_t2)

        Foo_dressed = Foo + 0.5 * lib.einsum("ic,jc->ij", Fov, t1)
        Fvv_dressed = Fvv - 0.5 * lib.einsum("lb,la->ab", Fov, t1)
        omega0 = self.get_omega0_factorized(X, Xvo, Xvo_t2, Fov, Y)
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
