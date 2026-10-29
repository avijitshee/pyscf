import os

from numpy.linalg import qr
from pyscf import lib, df
from pyscf.lib import logger
import numpy as np
from dataclasses import dataclass
import scipy.sparse.linalg
import scipy.special

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

    def kernel(self, t1, t2_act ,**kwargs):

        print('Starting low-level MPCC iteration...')
        try:
            func = self._kernels[self.kernel_type]
        except KeyError:
            raise ValueError(f'Unknown low-level kernel type: {self.kernel_type}')
        
        return func(t1, t2_act ,**kwargs)

    def _unfactorized_kernel(self, t1=None, t2=None, **kwargs):
        print('In unfactorized Kernel')

        #ll_method = kwargs.get('ll_method', self.ll_method)
        if self.ll_method == 'rpax':
            update_amps = self.update_amps_unfactorized_RPA
        elif self.ll_method == 'sylvester_laplace':
            update_amps = self.update_amps_sylvester_laplace
        elif self.ll_method == 'T1_transform':
            update_amps = self.update_amps_unfactorized
        else:
            raise ValueError(
                f"Unknown ll_method: {self.ll_method}. "
                "Use 'rpax', 'T1_transform', or 'sylvester_laplace'."
            )
        
        err = np.inf
        count = 0
        adiis = lib.diis.DIIS()

        e_corr = None

        while err > self.ll_con_tol and count < self.ll_max_its:

            res, e_corr, t1_new, t2_new = update_amps(t1, t2)
            if self.diis and self.ll_method == "sylvester_laplace":
                t1_new = self.run_diis(t1_new, adiis)
            elif self.diis:
                t1_new, t2_new = self.run_diis_full(t1_new, t2_new, adiis)
            else:
                t1_new, t2_new = t1_new, t2_new

            t1, t2 = t1_new, t2_new
            t1_new, t2_new = None, None  # free memory
            
            count += 1
            err = res
            # NOTE change this to logger!
            print(f"It {count}; correlation energy {e_corr:.6e}; residual {res:.6e}")

        del adiis
        self._e_corr = self.get_energy(t1, t2)
        self._e_tot = self.mf.e_tot + self._e_corr

        return t1, t2

    def update_amps_unfactorized(self, t1, t2):

        # Contractions
        Xoo, Xvo, X = self.get_X(t1)

        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        Ω = self.get_Ω_slow(X, Xvo, Foo, Fvv, Fov, t1, t2)

        res2 = self.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)
    
        ΔE = self.get_energy(t1, t2)

        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            Ω[np.ix_(act_particle, act_hole)] = 0.0
            res2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0

        res1 = Ω.T / self._eris.eia
        res2 = res2 / self._eris.D
        res = np.linalg.norm(res1) + np.linalg.norm(res2)

        t1 -= res1
        t2 -= res2

        return res, ΔE, t1, t2

    def update_amps_sylvester_laplace(self, t1, t2):
        """Update amplitudes using a Laplace-quadrature Sylvester T2 solve."""
        Xoo, Xvo, X = self.get_X(t1)
        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        Foo_eff, Fvv_eff = self.update_F(Foo.copy(), Fvv.copy(), Fov, t1)
        t2_ll = self.solve_t2_sylvester_laplace(Jvo, Foo_eff, Fvv_eff)

        omega = self.get_Ω_slow(X, Xvo, Foo, Fvv, Fov, t1, t2_ll)
        ΔE = self.get_energy(t1, t2)

        t2_act = []
        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            t2_act.append(
                t2[np.ix_(act_hole, act_hole, act_particle, act_particle)]
            )

        t2_new, omega = self.include_t2_active_dense(
            Foo_eff, Fvv_eff, Fov, t2_act, t2_ll, omega
        )

        res1 = omega.T / self._eris.eia
        res = np.linalg.norm(res1)

        t1 -= res1

        return res, ΔE, t1, t2_new

    def get_sylvester_intermediates(self, t1):
        """
        Build the intermediates used by Eq. T2_ampl_env_noDC.

        Returns
        -------
        Jvo : np.ndarray
            T1-transformed DF coupling with shape (naux, nvir, nocc).
        Foo : np.ndarray
            Effective occupied Fock block used by the Sylvester operator.
        Fvv : np.ndarray
            Effective virtual Fock block used by the Sylvester operator.
        Fov : np.ndarray
            Effective occupied-virtual Fock block, returned for verification
            against the existing unfactorized update_t2 implementation.
        """
        Xoo, Xvo, X = self.get_X(t1)
        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)
        Foo, Fvv = self.update_F(Foo, Fvv, Fov, t1)

        return Jvo, Foo, Fvv, Fov

    def solve_t2_sylvester_laplace(self, Jvo, Foo, Fvv, quad=None):
        """Solve Eq. T2_ampl_env_noDC with scalar denominator quadrature."""
        Foo = 0.5 * (Foo + Foo.T)
        Fvv = 0.5 * (Fvv + Fvv.T)

        eo, Uo = np.linalg.eigh(Foo)
        ev, Uv = np.linalg.eigh(Fvv)

        Jvo = lib.einsum("aA,Lai,iI->LAI", Uv, Jvo, Uo)
        denom = lib.direct_sum("A+B-I-J->IJAB", ev, ev, eo, eo)
        if np.any(denom <= 0.0):
            raise ValueError("Laplace Sylvester denominators must be positive")

        if quad is None:
            if self.ll_laplace_quad is not None:
                quad = self.ll_laplace_quad
            elif self.ll_laplace_quad_file is not None:
                self.ll_laplace_quad = laplace_quadrature.load(self.ll_laplace_quad_file)
                quad = self.ll_laplace_quad
            elif self.ll_laplace_root is not None:
                quad = laplace_quadrature.from_denominators(
                    self.ll_laplace_root,
                    denom,
                    self.ll_laplace_npoints,
                )
            else:
                raise ValueError(
                    "No Laplace quadrature configured. Provide ll_laplace_quad, "
                    "ll_laplace_quad_file, or ll_laplace_root."
                )
        interval_tol = 100.0 * np.finfo(float).eps * max(1.0, quad.ymax)
        if np.min(denom) < quad.ymin - interval_tol or np.max(denom) > quad.ymax + interval_tol:
            raise ValueError("Laplace quadrature interval does not cover denominators")

        inv_denom_laplace = np.zeros_like(denom)
        for exponent, weight in zip(quad.exponents, quad.weights):
            inv_denom_laplace += weight * np.exp(-exponent * denom)
        del denom

        t2 = -lib.einsum("LAI,LBJ,IJAB->IJAB", Jvo, Jvo, inv_denom_laplace)
        del inv_denom_laplace
        return lib.einsum("iI,jJ,aA,bB,IJAB->ijab", Uo, Uo, Uv, Uv, t2)

    def get_sylvester_laplace_factors(self, Jvo, Foo, Fvv, quad=None):
        """Build factorized Laplace amplitudes Y with t2 = -Y Y^T."""
        Foo = 0.5 * (Foo + Foo.T)
        Fvv = 0.5 * (Fvv + Fvv.T)

        eo, Uo = np.linalg.eigh(Foo)
        ev, Uv = np.linalg.eigh(Fvv)

        Jvo = lib.einsum("aA,Lai,iI->LAI", Uv, Jvo, Uo)
        denom = lib.direct_sum("A+B-I-J->IJAB", ev, ev, eo, eo)
        if np.any(denom <= 0.0):
            raise ValueError("Laplace Sylvester denominators must be positive")

        if quad is None:
            quad = self.get_sylvester_laplace_quadrature(denom)
        if np.any(quad.weights < 0.0):
            raise ValueError("factorized Laplace Sylvester requires nonnegative weights")
        interval_tol = 100.0 * np.finfo(float).eps * max(1.0, quad.ymax)
        if np.min(denom) < quad.ymin - interval_tol or np.max(denom) > quad.ymax + interval_tol:
            raise ValueError("Laplace quadrature interval does not cover denominators")
        del denom

        factors = np.empty(
            (Jvo.shape[0], quad.nlap, Jvo.shape[1], Jvo.shape[2]),
            dtype=Jvo.dtype,
        )
        for idx, (exponent, weight) in enumerate(zip(quad.exponents, quad.weights)):
            vo_scale = np.exp(-exponent * ev)
            oo_scale = np.exp(exponent * eo)
            Jvo_mu = np.sqrt(weight) * Jvo * vo_scale[None, :, None]
            Jvo_mu = Jvo_mu * oo_scale[None, None, :]
            factors[:, idx] = lib.einsum("aA,LAI,iI->Lai", Uv, Jvo_mu, Uo)

        return factors

    def get_sylvester_laplace_matrix_factors(
            self, Jvo, Foo, Fvv, quad=None, tol=1.0e-10, max_degree=200):
        """Build Laplace factors by applying matrix exponentials to Jvo.

        The Chebyshev matrix sequences T_k(Fvv_scaled) and T_k(Foo_scaled)
        depend only on the operator, not on the quadrature exponent.  They are
        therefore precomputed once (cost O(K * n_vir^3 + K * n_occ^3)) and
        reused for all nlap points.  Per quadrature point the cost reduces to
        a weighted sum of small matrices plus two BLAS dgemm calls, saving a
        factor of ~K compared to re-running the full Chebyshev recursion on the
        large (n_aux, n_vir, n_occ) tensor for every point.
        """
        t_total_start = time.time()

        if quad is None:
            ymin, ymax = self.get_sylvester_laplace_interval()
            quad = self.get_sylvester_laplace_quadrature_interval(ymin, ymax)
        if np.any(quad.weights < 0.0):
            raise ValueError("factorized Laplace Sylvester requires nonnegative weights")

        bounds_v = self._symmetric_spectral_bounds(Fvv)
        bounds_o = self._symmetric_spectral_bounds(Foo)

        n_aux, n_vir, n_occ = Jvo.shape
        nlap = quad.nlap

        lower_v, upper_v = bounds_v
        center_v = 0.5 * (upper_v + lower_v)
        radius_v = 0.5 * (upper_v - lower_v)

        lower_o, upper_o = bounds_o
        center_o = 0.5 * (upper_o + lower_o)
        radius_o = 0.5 * (upper_o - lower_o)

        # --- Step 1: precompute all Chebyshev coefficient vectors ---
        # Coefficients depend only on the exponent + scalar interval; cheap.
        all_coeffs_v = [
            self._chebyshev_exp_coefficients(-exp, center_v, radius_v, tol, max_degree)
            for exp in quad.exponents
        ]
        all_coeffs_o = [
            self._chebyshev_exp_coefficients(exp, center_o, radius_o, tol, max_degree)
            for exp in quad.exponents
        ]
        K_max_v = max(len(c) for c in all_coeffs_v)
        K_max_o = max(len(c) for c in all_coeffs_o)

        # --- Step 2: build Chebyshev matrix sequences once ---
        # M_v[k] = T_k(Fvv_scaled), shape (n_vir, n_vir)
        # M_o[k] = T_k(Foo_scaled), shape (n_occ, n_occ)
        # These are independent of the quadrature exponents.
        t_cheb_matrices = time.time()
        Fvv_scaled = (Fvv - center_v * np.eye(n_vir)) / radius_v
        M_v = np.empty((K_max_v, n_vir, n_vir), dtype=Jvo.dtype)
        M_v[0] = np.eye(n_vir, dtype=Jvo.dtype)
        if K_max_v > 1:
            M_v[1] = Fvv_scaled
            for k in range(2, K_max_v):
                M_v[k] = 2.0 * Fvv_scaled @ M_v[k - 1] - M_v[k - 2]

        Foo_scaled = (Foo - center_o * np.eye(n_occ)) / radius_o
        M_o = np.empty((K_max_o, n_occ, n_occ), dtype=Jvo.dtype)
        M_o[0] = np.eye(n_occ, dtype=Jvo.dtype)
        if K_max_o > 1:
            M_o[1] = Foo_scaled
            for k in range(2, K_max_o):
                M_o[k] = 2.0 * Foo_scaled @ M_o[k - 1] - M_o[k - 2]

        # Flatten last two dims for vectorised weighted sums
        M_v_flat = M_v.reshape(K_max_v, n_vir * n_vir)   # (K_max_v, n_vir^2)
        M_o_flat = M_o.reshape(K_max_o, n_occ * n_occ)   # (K_max_o, n_occ^2)
        print(f"  [get_sylvester_laplace_matrix_factors] Chebyshev matrix sequences "
              f"(K_v={K_max_v}, K_o={K_max_o}): {time.time() - t_cheb_matrices:.3f}s")

        # --- Step 3: apply exp matrices to Jvo for each quadrature point ---
        # Per point: two dgemm calls on the full (n_aux, n_vir, n_occ) tensor.
        # Left  action: exp_vv @ Jvo,   exp_vv: (n_vir, n_vir)
        # Right action: Jhat  @ exp_oo, exp_oo: (n_occ, n_occ)
        # Use explicit reshape to guarantee single BLAS dgemm per action.
        # Jvo: (n_aux, n_vir, n_occ) → (n_aux*n_occ, n_vir) for the left step.
        Jvo_2d = np.ascontiguousarray(Jvo.transpose(0, 2, 1)).reshape(n_aux * n_occ, n_vir)

        factors = np.empty((n_aux, nlap, n_vir, n_occ), dtype=Jvo.dtype)
        t_apply = time.time()
        for idx, (weight, cv, co) in enumerate(
            zip(quad.weights, all_coeffs_v, all_coeffs_o)
        ):
            # Build exp(-exponent * Fvv): (n_vir, n_vir) via vector-matrix dot
            exp_vv = (cv @ M_v_flat[:len(cv)]).reshape(n_vir, n_vir)

            # Left action: exp_vv @ Jvo  — single dgemm (n_aux*n_occ, n_vir)
            # Jvo_2d[L*n_occ+i, b] = Jvo[L, b, i]
            # result[L*n_occ+i, a] = sum_b Jvo[L, b, i] * exp_vv[a, b]
            Jhat_2d = Jvo_2d @ exp_vv.T              # (n_aux*n_occ, n_vir)

            # Build exp(exponent * Foo): (n_occ, n_occ)
            exp_oo = (co @ M_o_flat[:len(co)]).reshape(n_occ, n_occ)

            # Right action: Jhat @ exp_oo  — single dgemm (n_aux*n_vir, n_occ)
            Jhat = np.ascontiguousarray(
                Jhat_2d.reshape(n_aux, n_occ, n_vir).transpose(0, 2, 1)
            ).reshape(n_aux * n_vir, n_occ)           # (n_aux*n_vir, n_occ)
            result = (Jhat @ exp_oo).reshape(n_aux, n_vir, n_occ)

            factors[:, idx] = np.sqrt(weight) * result

        print(f"  [get_sylvester_laplace_matrix_factors] apply exp matrices "
              f"({nlap} points): {time.time() - t_apply:.3f}s")
        print(f"  [get_sylvester_laplace_matrix_factors] total: "
              f"{time.time() - t_total_start:.3f}s")

        del M_v_flat, M_o_flat, Jvo_2d, all_coeffs_v, all_coeffs_o
        return factors

    @staticmethod
    def _symmetric_spectral_bounds(matrix):
        """Return tight scalar spectral bounds without eigenvectors."""
        matrix = np.asarray(matrix)
        #symmetrize the matrix to avoid complex eigenvalues from numerical noise
        matrix = 0.5 * (matrix + matrix.T)
        if matrix.shape[0] == 1:
            value = float(matrix[0, 0])
            pad = max(1.0, abs(value)) * np.finfo(float).eps
            return value - pad, value + pad

        lower = float(
            scipy.sparse.linalg.eigsh(
                matrix, k=1, which="SA", return_eigenvectors=False
            )[0]
        )
        upper = float(
            scipy.sparse.linalg.eigsh(
                matrix, k=1, which="LA", return_eigenvectors=False
            )[0]
        )
        pad = 100.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
        return lower - pad, upper + pad

    @staticmethod
    def _chebyshev_exp_coefficients(scale, center, radius, tol, max_degree):
        prefactor = np.exp(scale * center)
        beta = scale * radius
        coeffs = [prefactor * scipy.special.iv(0, beta)]
        for degree in range(1, max_degree + 1):
            coeff = 2.0 * prefactor * scipy.special.iv(degree, beta)
            coeffs.append(coeff)
            if abs(coeff) <= tol * max(1.0, abs(coeffs[0])):
                return np.asarray(coeffs)
        raise RuntimeError(
            "Chebyshev exponential action did not converge within max_degree"
        )

    @classmethod
    def _chebyshev_exp_action_left(
        cls, matrix, rhs, scale, bounds, tol=1.0e-12, max_degree=200):
        lower, upper = bounds
        center = 0.5 * (upper + lower)
        radius = 0.5 * (upper - lower)
        coeffs = cls._chebyshev_exp_coefficients(
            scale, center, radius, tol, max_degree
        )

        inv_radius = 1.0 / radius
        c_over_r = center * inv_radius

        def apply_scaled(x):
            # matrix @ x broadcasts over the leading L dimension: (a,b) @ (L,b,i) -> (L,a,i)
            return np.matmul(matrix, x) * inv_radius - c_over_r * x

        t0 = rhs
        out = coeffs[0] * t0
        if len(coeffs) == 1:
            return out
        t1 = apply_scaled(t0)
        out = out + coeffs[1] * t1
        for coeff in coeffs[2:]:
            t2 = 2.0 * apply_scaled(t1) - t0
            out += coeff * t2
            t0, t1 = t1, t2
        return out

    @classmethod
    def _chebyshev_exp_action_right(
        cls, matrix, rhs, scale, bounds, tol=1.0e-12, max_degree=200):
        lower, upper = bounds
        center = 0.5 * (upper + lower)
        radius = 0.5 * (upper - lower)
        coeffs = cls._chebyshev_exp_coefficients(
            scale, center, radius, tol, max_degree
        )

        inv_radius = 1.0 / radius
        c_over_r = center * inv_radius

        def apply_scaled(x):
            # x @ matrix broadcasts over leading dims: (L,a,k) @ (k,i) -> (L,a,i)
            return np.matmul(x, matrix) * inv_radius - c_over_r * x

        t0 = rhs
        out = coeffs[0] * t0
        if len(coeffs) == 1:
            return out
        t1 = apply_scaled(t0)
        out = out + coeffs[1] * t1
        for coeff in coeffs[2:]:
            t2 = 2.0 * apply_scaled(t1) - t0
            out += coeff * t2
            t0, t1 = t1, t2
        return out

    def get_sylvester_laplace_interval(self):
        """Return denominator bounds from precomputed one-particle gaps."""
        eia = np.asarray(self._eris.eia)
        if np.any(eia <= 0.0):
            raise ValueError("Laplace Sylvester one-particle gaps must be positive")
        return 2.0 * float(np.min(eia)), 2.0 * float(np.max(eia))

    def get_sylvester_laplace_quadrature_interval(self, ymin, ymax):
        """Return the configured Laplace quadrature for a scalar interval."""
        if self.ll_laplace_quad is not None:
            quad = self.ll_laplace_quad
        elif self.ll_laplace_quad_file is not None:
            self.ll_laplace_quad = laplace_quadrature.load(self.ll_laplace_quad_file)
            quad = self.ll_laplace_quad
        elif self.ll_laplace_root is not None:
            self.ll_laplace_quad = laplace_quadrature.from_init_table(
                self.ll_laplace_root,
                ymin,
                ymax,
                self.ll_laplace_npoints,
            )
            quad = self.ll_laplace_quad
        else:
            raise ValueError(
                "No Laplace quadrature configured. Provide ll_laplace_quad, "
                "ll_laplace_quad_file, or ll_laplace_root."
            )

        interval_tol = 100.0 * np.finfo(float).eps * max(1.0, quad.ymax)
        if ymin < quad.ymin - interval_tol or ymax > quad.ymax + interval_tol:
            raise ValueError("Laplace quadrature interval does not cover denominators")
        return quad

    def get_sylvester_laplace_quadrature(self, denominators):
        """Return the configured Laplace quadrature for a denominator range."""
        if self.ll_laplace_quad is not None:
            return self.ll_laplace_quad
        if self.ll_laplace_quad_file is not None:
            self.ll_laplace_quad = laplace_quadrature.load(self.ll_laplace_quad_file)
            return self.ll_laplace_quad
        if self.ll_laplace_root is not None:
            self.ll_laplace_quad = laplace_quadrature.from_denominators(
                self.ll_laplace_root,
                denominators,
                self.ll_laplace_npoints,
            )
            return self.ll_laplace_quad
        raise ValueError(
            "No Laplace quadrature configured. Provide ll_laplace_quad, "
            "ll_laplace_quad_file, or ll_laplace_root."
        )


    def update_amps_unfactorized_RPA(self, t1, t2):

        # Contractions
        Xoo, Xvo, X = self.get_X(t1)

        Xvo_t2 = self.get_Xvo_t2(t2)
        Joo, Jvo, Jvv = self.get_J_RPA(Xoo, Xvo, Xvo_t2, t1)
        
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Xvo_t2)    

        Ω = self.get_Ω_slow_RPA(X, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2)

        res2 = self.update_t2_RPA(t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv, Xvo_t2)
    
        ΔE = self.get_energy(t1, t2)

        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            Ω[np.ix_(act_particle, act_hole)] = 0.0
            res2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0

        res1 = Ω.T / self._eris.eia
        res2 = res2 / self._eris.D
        res = np.linalg.norm(res1) + np.linalg.norm(res2)

        t1 -= res1
        t2 -= res2

        return res, ΔE, t1, t2


    def _factorized_kernel(self, t1=None, t2=None, **kwargs):

        res = np.inf
        count = 0
        adiis = lib.diis.DIIS()

        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            t2_act = t2[np.ix_(act_hole, act_hole, act_particle, act_particle)]

        e_corr = None
        while res > self.ll_con_tol and count < self.ll_max_its:

            res, t1_it, Δt2s_o, Δt2s_v, Y = self.update_amps_factorized(t1, t2_act, self._Y, **kwargs)
            if self.diis:
                t1_it  = self.run_diis(t1_it, adiis)

            t1 = t1_it
            self._Y = Y
            
            count += 1
            # NOTE change this to logger!
            print(f"It {count}; residual {res:.6e}")

        del adiis
        # NOTE non-iterative N^5 cost!
        t2 = self.get_t2(Y, t2_act, Δt2s_o, Δt2s_v)
        self._e_corr = self.get_energy(t1, t2) 

        return t1, t2

    def _sylvester_laplace_factorized_kernel(self, t1=None, t2=None, **kwargs):
        """Low-memory Laplace Sylvester kernel that iterates with Y factors."""
        res = np.inf
        count = 0
        adiis = lib.diis.DIIS()
        Δt2s_o = []
        Δt2s_v = []
        self._t2_full = None

        t2_act = []
        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            t2_act.append(t2[np.ix_(act_hole, act_hole, act_particle, act_particle)])

        while res > self.ll_con_tol and count < self.ll_max_its:
            res, t1_new, Δt2s_o, Δt2s_v, Y = (
                self.update_amps_sylvester_laplace_factorized(
                    t1, t2_act, **kwargs
                )
            )
            if self.diis:
                t1_new = self.run_diis(t1_new, adiis)

            t1 = t1_new
            self._Y = Y

            count += 1
            print(f"It {count}; residual {res:.6e}")

        del adiis
        if self._t2_full is None:
            t2 = self.get_t2_factorized_laplace(Y, t2_act, Δt2s_o, Δt2s_v)
        else:
            t2 = self._t2_full
        self._e_corr = self.get_energy(t1, t2)

        return t1, t2

    def _sylvester_laplace_factorized_noniterative_kernel(self, t1=None, t2=None, **kwargs):
        # free init_amps allocations that this kernel does not use
        self._Y = None
        self._t2 = None

        t_kernel_start = time.time()

        res = np.inf
        count = 0
        adiis = lib.diis.DIIS()
        Δt2s_o = []
        Δt2s_v = []
        self._t2_full = None

        t2_act = []
        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            t2_act.append(t2[np.ix_(act_hole, act_hole, act_particle, act_particle)])

        #if file exists, load Y, Δt2s_o, Δt2s_v from disk for later use:
        t_load_start = time.time()
        if os.path.exists("Y.npy"):
            Y_old = np.load("Y.npy")
        if os.path.exists("Δt2s_o.npy") and os.path.exists("Δt2s_v.npy"):    
            Δt2s_o_old = np.load("Δt2s_o.npy")
            Δt2s_v_old = np.load("Δt2s_v.npy")
            print(f"Loaded Y, Δt2s_o, Δt2s_v from disk in {time.time() - t_load_start:.3f}s")
        else:
            Y_old = None
            Δt2s_o_old = None
            Δt2s_v_old = None

        t_iter_total = 0.0
        while res > self.ll_con_tol and count < self.ll_max_its:
            #update t1
            t_iter_start = time.time()
            res, t1_new = self.update_t1(t1, Y_old, Δt2s_o_old, Δt2s_v_old)
            
            if self.diis:
                t1_new = self.run_diis(t1_new, adiis) 

            t1 = t1_new 
            count += 1
            t_iter = time.time() - t_iter_start
            t_iter_total += t_iter
            print(f"It {count}; residual {res:.6e}; iter time {t_iter:.3f}s")

        print(f"T1 iterations total time: {t_iter_total:.3f}s ({count} iterations)")

        del adiis
        Y_old = Δt2s_o_old = Δt2s_v_old = None
        #get Y, Δt2s_o, Δt2s_v
        t_Y_start = time.time()
        Y, Δt2s_o, Δt2s_v = self.update_Y_Dt2(t1, t2_act)
        print(f"update_Y_Dt2 time: {time.time() - t_Y_start:.3f}s")

        #dump Y, Δt2s_o, Δt2s_v to disk for later use:
        t_save_start = time.time()
        if Y is not None:
           np.save("Y.npy", Y)
        if Δt2s_o is not None and Δt2s_v is not None:   
           np.save("Δt2s_o.npy", Δt2s_o)
           np.save("Δt2s_v.npy", Δt2s_v)
           print(f"Save Y, Δt2s_o, Δt2s_v to disk time: {time.time() - t_save_start:.3f}s")

        t_t2_start = time.time()
        if self._t2_full is None:
            t2 = self.get_t2_factorized_laplace(Y, t2_act, Δt2s_o, Δt2s_v)
        else:
            t2 = self._t2_full
        print(f"T2 reconstruction time: {time.time() - t_t2_start:.3f}s")

        print(f"_sylvester_laplace_factorized_noniterative_kernel total time: {time.time() - t_kernel_start:.3f}s")

        return t1, t2

    def update_amps_sylvester_laplace_factorized(self, t1, t2_act, **kwargs):
        """Update T1 while keeping Laplace Sylvester T2 in factorized form."""
        Xoo, Xvo, X = self.get_X(t1)
        _, Jvo = self.get_J(Xoo, Xvo, t1)
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        Foo_eff, Fvv_eff = self.update_F(Foo.copy(), Fvv.copy(), Fov, t1)
        Y = self.get_sylvester_laplace_matrix_factors(Jvo, Foo_eff, Fvv_eff)

        Ω = self.get_Ω_sylvester_laplace_factorized(
            X, Xvo, Foo, Fvv, Fov, t1, Y) 

        Δt2s_o, Δt2s_v = self.include_t2_active_factorized_laplace(
            Foo_eff, Fvv_eff, Fov, t2_act, Y)

        Ω =self.get_t1_correction_Dt2(Fov, Δt2s_o, Δt2s_v, Ω)

        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            Ω[np.ix_(act_particle, act_hole)] = 0.0

        self._t2_full = None

        res1 = Ω.T / self._eris.eia
        t1 -= res1

        return np.linalg.norm(res1), t1, Δt2s_o, Δt2s_v, Y

    
    def update_t1(self, t1, Y=None , Δt2s_o=None, Δt2s_v=None):

       Xoo, Xvo, X = self.get_X(t1)
       _, Jvo = self.get_J(Xoo, Xvo, t1)
       Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)
       
       if Y is not None and Δt2s_o is not None and Δt2s_v is not None:
          Ω = self.get_Ω_sylvester_laplace_factorized(
               X, Xvo, Foo, Fvv, Fov, t1, Y, Δt2s_o, Δt2s_v) 
       else:
          Ω = self.get_Ω_sylvester_laplace_factorized(
               X, Xvo, Foo, Fvv, Fov, t1) 
       
       for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            Ω[np.ix_(act_particle, act_hole)] = 0.0

       res1 = Ω.T / self._eris.eia
       t1 -= res1
       return np.linalg.norm(res1), t1

    def update_Y_Dt2(self, t1, t2_act):
        """Update T1 while keeping Laplace Sylvester T2 in factorized form."""
        Xoo, Xvo, X = self.get_X(t1)
        _, Jvo = self.get_J(Xoo, Xvo, t1)
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        Foo_eff, Fvv_eff = self.update_F(Foo.copy(), Fvv.copy(), Fov, t1)
        Y = self.get_sylvester_laplace_matrix_factors(Jvo, Foo_eff, Fvv_eff)

        #skip this if norm of t2_act is small
        if np.linalg.norm(t2_act) < 1e-8:
            return Y, None, None            
        else:
            Δt2s_o, Δt2s_v = self.include_t2_active_factorized_laplace(
                 Foo_eff, Fvv_eff, Fov, t2_act, Y)
            return Y, Δt2s_o, Δt2s_v 

    def get_Ω_sylvester_laplace_factorized(self, X, Xvo, Foo, Fvv, Fov, t1, Y=None, Δt2s_o=None, Δt2s_v=None):
        """Evaluate Omega for factorized amplitudes with t2 = -Y Y^T."""
        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy()

        Foo_tmp += lib.einsum("ic,jc->ij", Fov, t1) * 0.5
        Fvv_tmp -= lib.einsum("lb,la->ab", Fov, t1) * 0.5

        Ω = self._eris.fov.T.copy()

        Ω -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω += lib.einsum("Lai,L->ai", self._eris.Lvo, X)

        Ω += lib.einsum("ib,ab->ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki->ai", t1, Foo_tmp)

        if Y is not None:
           Ω_temp = lib.einsum("LRjb,bj->LR", Y, Fov)
           Ω -= 2.0 * lib.einsum("LR,LRai->ai", Ω_temp, Y)

           Ω_temp = lib.einsum("LRbi,jb->LRij", Y, Fov)
           Ω += lib.einsum("LRij,LRaj->ai", Ω_temp, Y)

        if Δt2s_o is not None and Δt2s_v is not None:
           Ω = self.get_t1_correction_Dt2(Fov, Δt2s_o, Δt2s_v, Ω)

        return Ω

    def get_t1_correction_Dt2(self, Fov, Δt2s_o, Δt2s_v, Ω):
        """Evaluate the T1 correction from the active T2 change."""
     
        for k, frag in enumerate(self.frags): 
            act_hole = frag[0]
            inact_hole = np.delete(range(self.nocc), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(self.nvir), act_particle)

            t2_antisym = 2.0 * Δt2s_o[k] - np.transpose(Δt2s_o[k], (0, 1, 3, 2))
            Ω[np.ix_(act_particle, inact_hole)] += lib.einsum(
                    "Ijab,jb->aI",
                    t2_antisym,
                    Fov[np.ix_(act_hole, act_particle)],
                ) 

            t2_antisym = 2.0 * Δt2s_v[k] - np.transpose(Δt2s_v[k], (1, 0, 2, 3))
            Ω[np.ix_(inact_particle, act_hole)] += lib.einsum(
                    "ijAb,jb->Ai",
                    t2_antisym,
                    Fov[np.ix_(act_hole, act_particle)],
               )   
        return Ω


    def include_t2_active_factorized_laplace(self, Foo, Fvv, Fov, t2_act, Y, tol=None, count_tol=None):
        """Solve the Eq:T2_error boundary correction without building T2_LL."""
        print(f'Computing active t2-correction ...')

        print("value of tol:", tol, self.ll_active_t2_tol) 
        if tol is None:
            tol = self.ll_active_t2_tol
        else:
            tol = min(tol, self.ll_active_t2_tol)
        if count_tol is None:
            count_tol = self.ll_active_t2_max_its

        Δt2s_o = []
        Δt2s_v = []

        # Try to load previous Δt2s from disk to warm-start GMRES (x0)
        _x0_o_prev = None
        _x0_v_prev = None
        if os.path.exists("Δt2s_o.npy") and os.path.exists("Δt2s_v.npy"):
            try:
                _x0_o_prev = np.load("Δt2s_o.npy", allow_pickle=True)
                _x0_v_prev = np.load("Δt2s_v.npy", allow_pickle=True)
                print("Loaded Δt2s_o, Δt2s_v from disk for GMRES warm start")
            except Exception as e:
                print(f"Could not load Δt2s warm-start files: {e}")
                _x0_o_prev = None
                _x0_v_prev = None

        n_aux, n_rank, n_vir, n_occ = Y.shape
        for k, frag in enumerate(self.frags):
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)

            δt2 = -lib.einsum(
                "LRai,LRbj->ijab",
                Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)],
                Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)],
            )
            Δt2_active = t2_act[k] - δt2
            del δt2

            shape_o = (
                len(inact_hole),
                len(act_hole),
                len(act_particle),
                len(act_particle),
            )
            shape_v = (
                len(act_hole),
                len(act_hole),
                len(inact_particle),
                len(act_particle),
            )
            size_o = int(np.prod(shape_o))
            size_v = int(np.prod(shape_v))
            size = size_o + size_v

            Foo_aa = Foo[np.ix_(act_hole, act_hole)]
            Foo_ii = Foo[np.ix_(inact_hole, inact_hole)]
            Foo_ai = Foo[np.ix_(act_hole, inact_hole)]
            Fvv_aa = Fvv[np.ix_(act_particle, act_particle)]
            Fvv_ii = Fvv[np.ix_(inact_particle, inact_particle)]
            Fvv_ia = Fvv[np.ix_(inact_particle, act_particle)]

            D_o = self._eris.D[np.ix_(
                inact_hole, act_hole, act_particle, act_particle
            )]
            D_v = self._eris.D[np.ix_(
                act_hole, act_hole, inact_particle, act_particle
            )]

            def split(vector):
                if size_o:
                    Δt2_o = vector[:size_o].reshape(shape_o)
                else:
                    Δt2_o = np.zeros(shape_o, dtype=Y.dtype)
                if size_v:
                    Δt2_v = vector[size_o:].reshape(shape_v)
                else:
                    Δt2_v = np.zeros(shape_v, dtype=Y.dtype)
                return Δt2_o, Δt2_v

            def pack_scaled(res_o, res_v):
                pieces = []
                if size_o:
                    pieces.append((res_o / D_o).ravel())
                if size_v:
                    pieces.append((res_v / D_v).ravel())
                if not pieces:
                    return np.empty(0, dtype=Y.dtype)
                return np.concatenate(pieces)

            def boundary_residual(Δt2_o, Δt2_v):
                res_o = np.zeros(shape_o, dtype=Y.dtype)
                res_v = np.zeros(shape_v, dtype=Y.dtype)

                if size_o:
                    res_o += lib.einsum("bc,Ijac->Ijab", Fvv_aa, Δt2_o)
                    res_o += lib.einsum("ac,Ijcb->Ijab", Fvv_aa, Δt2_o)
                    res_o -= lib.einsum("KI,Kjab->Ijab", Foo_ii, Δt2_o)
                    res_o -= lib.einsum("kj,Ikab->Ijab", Foo_aa, Δt2_o)

                if size_v:
                    res_v += lib.einsum("bc,ijAc->ijAb", Fvv_aa, Δt2_v)
                    res_v += lib.einsum("AC,ijCb->ijAb", Fvv_ii, Δt2_v)
                    res_v -= lib.einsum("ki,kjAb->ijAb", Foo_aa, Δt2_v)
                    res_v -= lib.einsum("kj,ikAb->ijAb", Foo_aa, Δt2_v)

                return res_o, res_v

            source_o = np.zeros(shape_o, dtype=Y.dtype)
            source_v = np.zeros(shape_v, dtype=Y.dtype)
            if size_o:
                source_o -= lib.einsum(
                    "kI,kjab->Ijab", Foo_ai, Δt2_active
                )
            if size_v:
                source_v += lib.einsum(
                    "Ac,jibc->ijAb", Fvv_ia, Δt2_active
                )

            rhs = -pack_scaled(source_o, source_v)
            rhs_norm = np.linalg.norm(rhs)

            if size and rhs_norm > tol:
                # Build GMRES initial guess (x0) from previously saved Δt2s
                x0 = None
                if (
                    _x0_o_prev is not None
                    and _x0_v_prev is not None
                    and k < len(_x0_o_prev)
                    and k < len(_x0_v_prev)
                ):
                    prev_o = np.asarray(_x0_o_prev[k])
                    prev_v = np.asarray(_x0_v_prev[k])
                    if prev_o.shape == shape_o and prev_v.shape == shape_v:
                        x0_pieces = []
                        if size_o:
                            x0_pieces.append(prev_o.ravel())
                        if size_v:
                            x0_pieces.append(prev_v.ravel())
                        if x0_pieces:
                            x0 = np.concatenate(x0_pieces).astype(Y.dtype)
                            print(f"    Fragment {k}: using warm-start x0 from Δt2s .npy files")

                operator = scipy.sparse.linalg.LinearOperator(
                    (size, size),
                    matvec=lambda vector: pack_scaled(
                        *boundary_residual(*split(vector))
                    ),
                    dtype=Y.dtype,
                )
                residual_history = []
                # Use lgmres (pure Python) instead of gmres: the Fortran-backed
                # gmres overflows its 32-bit workspace-size check once `size`
                # grows large, regardless of the tol/atol kwargs used.
                correction, info = scipy.sparse.linalg.lgmres(
                    operator,
                    rhs,
                    x0=x0,
#                    tol=min(1.0e-6, tol),
                    tol=1.0e-8, 
                    inner_m=min(size, 50),
                    maxiter=count_tol,
                    callback=residual_history.append,
                )
                Δt2_o, Δt2_v = split(correction)
                res_o, res_v = boundary_residual(Δt2_o, Δt2_v)
                acc = np.linalg.norm(pack_scaled(
                    res_o + source_o, res_v + source_v
                ))
                if info != 0:
                    raise RuntimeError(
                        "Factorized active T2 correction did not converge: "
                        f"info={info}, residual={acc:.3e}, target={tol:.3e}"
                    )
                print(
                    f'    GMRES T2 correction converged in '
                    f'{len(residual_history)} iterations with final accuracy '
                    f'{acc:.2e}'
                )
            else:
                Δt2_o = np.zeros(shape_o, dtype=Y.dtype)
                Δt2_v = np.zeros(shape_v, dtype=Y.dtype)
                acc = rhs_norm
                print(
                    f'    T2 correction already converged with final accuracy '
                    f'{acc:.2e}'
                )

            Δt2s_o.append(Δt2_o)
            Δt2s_v.append(Δt2_v)

        del _x0_o_prev, _x0_v_prev
        return Δt2s_o, Δt2s_v

    def update_amps_factorized(self, t1, t2_act, Y, **kwargs):
        """
        Following Table XXX in Future Paper
        """
        # Setp 1 
        Xoo, Xvo, X = self.get_X(t1)

        # Step 2
        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
       
        # Step 3
        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo)

        # Step 4 
        Ω = self.get_Ω(X, Xvo, Foo, Fvv, Fov, t1, Y)

        # NOTE discuss the grouping of actions, 5-7/8 seems to be fit for one step
        # Step 5 
        Foo, Fvv = self.update_F(Foo, Fvv, Fov, t1)

        # Step 6 & 7
        D, Uoo, Uvv = self.get_D(Foo, Fvv, **kwargs)

        # NOTE Can we transform D insead of J, that way we don't have to transform Y back!
        

        # Step 8
        Jvo = self.update_J(Jvo, Uoo, Uvv)

        # Step 9
        Y, Yt = self.update_Y_backup(Joo, Jvo, D, Uvv, Uoo, Y)

        # Step 10 & 11
        Δt2s_o, Δt2s_v, Ω = self.include_t2_active(Foo, Fvv, Fov, t2_act, Y, Ω)
       
        res = Ω.T / self._eris.eia
        t1 -= res

        # NOTE Checking energy convergence, remove energy computation later!
        t2 = self.get_t2(Y, t2_act, Δt2s_o, Δt2s_v)
        e_corr = self.get_energy(t1, t2) 
        print(f'    CC2 correlation energy: {e_corr}')


        return np.linalg.norm(res), t1, Δt2s_o, Δt2s_v, Y
    
    @dataclass
    class _IMDS_T1:
       oo: np.ndarray
       Jvv: np.ndarray
       Jov: np.ndarray
       Foo: np.ndarray
       Fvv: np.ndarray
       Fov: np.ndarray
       Xoo: np.ndarray
       Xvo: np.ndarray
    
    def get_t1_imds(self, t1):

        X, Xoo, Xvo = self.get_X(t1)

        #Joo, Jvo, Jvv = self.get_J_all(Xoo, Xvo, t1)
        Joo, Jvo = self.get_J(Xoo, Xvo, t1)
        Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) 

        Foo, Fvv, Fov = self.get_F(t1, X, Xoo, Xvo, Jvo)

        #store Joo, Jvo, Foo etc..
        imds_t1 = self._IMDS_T1(Joo=Joo, Jvv=Jvv, Jov=Jvo, Foo=Foo, Fvv=Fvv, Fov=Fov, Xoo=Xoo, Xvo=Xvo)
        return imds_t1
    
    def run_diis_full(self, t1, t2, adiis):

        def amplitudes_to_vector_full(t1, t2, out=None):
            nov = self.nocc * self.nvir
            size = nov + nov * (nov + 1) // 2
            vector = np.ndarray(size, t1.dtype, buffer=out)
            vector[:nov] = t1.ravel()
            lib.pack_tril(t2.transpose(0, 2, 1, 3).reshape(nov, nov), out=vector[nov:])
            return vector

        def vector_to_amplitudes_full(vector):
            nov = self.nocc * self.nvir
            t1 = vector[:nov].copy().reshape((self.nocc, self.nvir))
            # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
            t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
            t2 = t2.reshape(self.nocc, self.nvir, self.nocc, self.nvir).transpose(
                0, 2, 1, 3
            )
            return t1, np.asarray(t2, order="C")

        vec = amplitudes_to_vector_full(t1, t2)
        t1, t2 = vector_to_amplitudes_full(adiis.update(vec))

        return t1, t2

    def run_diis(self, t1, adiis):

        def amplitudes_to_vector(t1):
            nov = t1.shape[0] * t1.shape[1]
            vector = t1.ravel()
            return vector

        def vector_to_amplitudes(vector):
            nov = self.nocc * self.nvir
            t1 = vector[:nov].copy().reshape((self.nocc, self.nvir))
            return t1

        vec = amplitudes_to_vector(t1)
        t1 = vector_to_amplitudes(adiis.update(vec))

        return t1

    def run_diis_Δt2(self, t2_in, adiis):

        def amplitudes_to_vector(t2):
            return t2.ravel()

        def vector_to_amplitudes_shapes(vector, shapes):
            return vector.reshape(shapes)

        vec2amp = partial(vector_to_amplitudes_shapes, shapes = t2_in.shape)
        vec = amplitudes_to_vector(t2_in)

        return vec2amp(adiis.update(vec))

    def get_X(self, t1):

        Xvo = lib.einsum("Lab,ib->Lai", self._eris.Lvv, t1)
        Xoo = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        X = lib.einsum("Lia,ia->L", self._eris.Lov, t1)*2.0

        return Xoo, Xvo, X

    def add_t2_to_fock(self, Fvv, Foo, Xvo_t2):    

        Foo += lib.einsum("Lie,Lej->ij", self._eris.Lov,Xvo_t2)
        Fvv -= lib.einsum("Lmb,Lam->ab",self._eris.Lov,Xvo_t2)

        return Foo, Fvv


    def get_Xvo_t2(self, t2):

        t2_antisym = 2.0*t2 - t2.transpose(0, 1, 3, 2)

        Xvo_t2 = lib.einsum("Lkc,ikac ->Lai", self._eris.Lov, t2_antisym)

        return Xvo_t2


    def get_J(self, Xoo, Xvo, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
        )

        return Joo, Jvo


    def get_J_RPA(self, Xoo, Xvo, Xvo_t2, t1):

        Joo = Xoo + self._eris.Loo
        Jvo = (
            Xvo + Xvo_t2 + self._eris.Lvo - lib.einsum("Lji,ja->Lai", Joo, t1)
        )

        Jvv = self._eris.Lvv - lib.einsum("Lkb,ka->Lab", self._eris.Lov, t1) #we don't need this here  

        return Joo, Jvo, Jvv



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

        return Foo, Fvv, Fov

    # FIXME There is substantial code overlap with get_Ω
    def get_Ω_slow(self, X, Xvo, Foo, Fvv, Fov, t1, t2):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)*0.5
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)*0.5 

        Ω = self._eris.fov.T.copy()

        Ω -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω += lib.einsum("Lai,L->ai", self._eris.Lvo, X)


        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)

        t2_antisym = 2.0*t2 - np.transpose(t2, (0, 1, 3, 2))
        Ω += lib.einsum("ijab,jb->ai", t2_antisym, Fov)

        del Foo_tmp, Fvv_tmp
        
        return Ω 


    def get_Ω_slow_RPA(self, X, Xvo, Xvo_t2, Foo, Fvv, Fov, t1, t2):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)*0.5
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)*0.5 

        Ω = self._eris.fov.T.copy()

        Ω -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω -= lib.einsum("Laj,Lji->ai", Xvo_t2, self._eris.Loo) #new
        Ω += lib.einsum("Lai,L->ai", self._eris.Lvo, X)

        Ω += lib.einsum("Lae,Lei->ai", self._eris.Lvv, Xvo_t2) #new


        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)

        t2_antisym = 2.0*t2 - np.transpose(t2, (0, 1, 3, 2))
        Ω += lib.einsum("ijab,jb->ai", t2_antisym, Fov)

        del Foo_tmp, Fvv_tmp
        
        return Ω 


    def update_t2(self, t2, Jvo, Foo, Fvv, Fov, t1):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)
        
        tmp  = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("mi,mjab->ijab", Foo_tmp, t2)
        res2 = tmp + tmp.transpose(1,0,3,2)
        res2 += lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

        return res2

    def _svd_factorize_slice(self, mat, tol):

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        if s.size == 0:
            return u[:, :0], vh[:0]

        cutoff = tol * s[0]
        rank = np.count_nonzero(s > cutoff)
        if rank == 0:
            rank = 1

        return u[:, :rank] * s[:rank], vh[:rank]

    def _contract_n3v3(self, Joo, Jvv, t2):

        tol = self.ll_low_rank_tol
        if tol is None:
            return lib.einsum("Lmj,Lbe,imae->ijab", Joo, Jvv, t2, optimize=True)

        naux = Joo.shape[0]
        nocc = Joo.shape[1]
        nvir = Jvv.shape[1]

        contracted = np.zeros((nocc, nocc, nvir, nvir), dtype=t2.dtype)
        rank_oo = []
        rank_vv = []

        print(f"Performing low-rank factorization of Joo and Jvv with tol={tol:.2e} for N3V3 contraction...")

        for aux_idx in range(naux):
            left_oo, right_oo = self._svd_factorize_slice(Joo[aux_idx], tol)
            left_vv, right_vv = self._svd_factorize_slice(Jvv[aux_idx], tol)

            rank_oo.append(left_oo.shape[1])
            rank_vv.append(left_vv.shape[1])

            tmp_lr = lib.einsum("mp,qe,imae->iapq", left_oo, right_vv, t2, optimize=True)
            contracted += lib.einsum("pj,bq,iapq->ijab", right_oo, left_vv, tmp_lr, optimize=True)

        avg_rank_oo = float(np.mean(rank_oo))
        avg_rank_vv = float(np.mean(rank_vv))

        print(f"rank-reduced N3V3 contraction enabled: avg rank(Joo)={avg_rank_oo:.2f} avg rank(Jvv)={avg_rank_vv:.2f}")

        #logger.debug1(
        #    self,
        #    "rank-reduced N3V3 contraction enabled: avg rank(Joo)=%.2f avg rank(Jvv)=%.2f",
        #    avg_rank_oo,
        #    avg_rank_vv,
        #)
        return contracted


    def update_t2_RPA(self, t2, Jvo, Foo, Fvv, Fov, t1, Joo, Jvv, Xvo_t2):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)
        
        tmp  = lib.einsum("bc,ijac->ijab", Fvv_tmp, t2)
        tmp -= lib.einsum("mi,mjab->ijab", Foo_tmp, t2)

## N3V3

        #W_jebm = lib.einsum("Lmj, Lbe -> mbje", Joo, Jvv)
        #tmp -= lib.einsum("mbje, imae -> ijab", W_jebm, t2)

        tmp -= self._contract_n3v3(Joo, Jvv, t2)


#       W_jema = lib.einsum("Lmj, Lae -> maje", Joo, Jvv)
#       tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2) 

        res2 = tmp + tmp.transpose(1,0,3,2)
        res2 += lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)

    #    Waebf = lib.einsum("Lae, Lbf -> abef", Jvv, Jvv)
    #    res2 += lib.einsum("abef, ijef -> ijab", Waebf, t2)

    #    Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo, Joo)
    #    res2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2) 

#       res2 += lib.einsum("Lai,Lbj->ijab", Xvo_t2, Jvo)
#       res2 += lib.einsum("Lai,Lbj->ijab", Jvo, Xvo_t2)

        return res2


    def get_Ω(self, X, Xvo, Foo, Fvv, Fov, t1, Y):

        Foo_tmp = Foo.copy()
        Fvv_tmp = Fvv.copy() 

        Foo_tmp += lib.einsum("ic,jc->ij",Fov,t1)*0.5
        Fvv_tmp -= lib.einsum("lb,la->ab",Fov,t1)*0.5 

        Ω = self._eris.fov.T.copy()

        Ω -= lib.einsum("Laj,Lji->ai", Xvo, self._eris.Loo)
        Ω += lib.einsum("Lai,L->ai", self._eris.Lvo, X)

        Ω += lib.einsum("ib,ab -> ai", t1, Fvv_tmp)
        Ω -= lib.einsum("ka,ki -> ai", t1, Foo_tmp)
      
        # Here adjust to only update the environment part 
        Ω_temp = lib.einsum("LRjb, bj -> LR", Y, Fov)
        Ω += 2* lib.einsum("LR, LRai -> ai", Ω_temp, Y) 

        Ω_temp = lib.einsum("LRbi, jb -> LRij", Y, Fov)
        Ω -= lib.einsum("LRij, LRaj", Ω_temp, Y)

        del Foo_tmp, Fvv_tmp, Ω_temp

        return Ω


    def t2_transform_quadratic(self,t2):

        Vnemf = lib.einsum("Lne, Lmf -> nmef", self._eris.Lov, self._eris.Lov)
        #I^je_mb
        Imbje = lib.einsum("nmef, jnfb -> mbje", Vnemf, t2) # #nf should be ii, ia, ai types

        return Imbje


    def update_F(self, Foo, Fvv, Fov, t1):

        Fvv = Fvv - lib.einsum("ic,ib -> bc", Fov, t1) 
        Foo = Foo + lib.einsum("kc,ic -> ki", Fov, t1)
 
        return Foo, Fvv

    def get_D(self, Foo, Fvv, **kwargs):

        Foo = 0.5 * (Foo + Foo.T)
        Fvv = 0.5 * (Fvv + Fvv.T)

        #print(f"Symmetry in Foo: {np.linalg.norm(Foo - Foo.T)}")
        #print(f"Symmetry in Fvv: {np.linalg.norm(Fvv - Fvv.T)}")

        e_oo, Uoo = np.linalg.eigh(Foo)
        e_vv, Uvv = np.linalg.eigh(Fvv)
        eia = lib.direct_sum("-i+a->ia", e_oo, e_vv)
      
        if 'chol_tol' in kwargs:
            chol_tol = kwargs['chol_tol']
        else:
            chol_tol = None
       
        if 'chol_rank' in kwargs:
            chol_rank = kwargs['chol_rank']
        else:
            chol_rank = None
        
        D = mpcc_tools.piv_chol_tensor(eia, tol = chol_tol, rank = chol_rank)

        return D, Uoo, Uvv

    def update_J(self, Jvo, Uoo, Uvv):

        Jvo = np.einsum("ab, ij, Lai -> Lbj", Uvv, Uoo, Jvo)

        return Jvo

    def update_Y(self, Jvo, D, Uvv, Uoo):

        dt = np.einsum('jbr, ij, ab', D, Uoo, Uvv)
        return Jvo[:, None, :, :] *D.transpose(2, 1, 0)[None, :, :, :]

    def update_Y_backup(self, Joo, Jvo, D, Uvv, Uoo, Y):

        st = time.time()
        Yt = Jvo[:, None, :, :] *D.transpose(2, 1, 0)[None, :, :, :]
        print(f'HP elapsed time: {time.time() - st}')

        # NOTE REMOVE THE CHECK IF WE ARE HAPPY!!!
        if False:
            t2_app = lib.einsum("LRai, LRbj -> ijab", Yt, Yt)    
            
            fro_rel = np.linalg.norm(t2 - t2_app) / np.linalg.norm(t2)
            max_abs = np.max(np.abs(t2 - t2_app))

            print(f'Test in update_Y:')
            print(f"4th-order relative Frobenius error: {fro_rel:.3e}")
            print(f"4th-order max abs entry error:     {max_abs:.3e}")

        st = time.time()
        Y = np.einsum("LRbj, ab, ij -> LRai", Yt, Uvv, Uoo, optimize = True)
        print(f'EINSUM elapsed time: {time.time() - st}')
        return Y, Yt

    def include_t2_active(self, Foo, Fvv, Fov, t2_act, Y, Ω, tol = 1e-6, count_tol = 1000):
       
        print(f'Computing active t2-correction ...')   
        Δt2s_o = [] 
        Δt2s_v = [] 

        n_aux, n_rank, n_vir, n_occ = Y.shape
        for k, frag in enumerate(self.frags):

            # FIXME once fragmentation is assigned, compute these once!!!
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)
 
            eia_o = lib.direct_sum("Ia+jb->Ijab", 
                                   self._eris.eia[np.ix_(inact_hole, act_particle)], 
                                   self._eris.eia[np.ix_(act_hole, act_particle)])
            eia_v = lib.direct_sum("iA+jb->ijAb", 
                                   self._eris.eia[np.ix_(act_hole, inact_particle)], 
                                   self._eris.eia[np.ix_(act_hole, act_particle)])

            # NOTE This is the truly iterative part
            # Step 10  
            Ω[np.ix_(act_particle, act_hole)] = 0.0

            δt2 = -lib.einsum("LRai, LRbj -> ijab", 
                               Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)], 
                               Y[np.ix_(range(n_aux), range(n_rank), act_particle, act_hole)])    
            Δt2 = t2_act[k] - δt2 
  
            Δt2_o = -lib.einsum('kI, kjab -> Ijab ',Foo[np.ix_(act_hole, inact_hole)], Δt2)
        #   Δt2_o -=  lib.einsum('kJ, ikab -> Jiab ',Foo[np.ix_(act_hole, inact_hole)], Δt2)
            Δt2_o -=  lib.einsum('kJ, kiba -> Jiba ',Foo[np.ix_(act_hole, inact_hole)], Δt2) 

            Δt2_v = lib.einsum('Ac, ijcb -> ijAb ',Fvv[np.ix_(inact_particle, act_particle)], Δt2)
            Δt2_v += lib.einsum('Bc, ijac -> ijBa ',Fvv[np.ix_(inact_particle, act_particle)], Δt2)
          # Δt2_v += lib.einsum('Bc, jica -> jiBa ',Fvv[np.ix_(inact_particle, act_particle)], Δt2)

            #calculate the norm of Δt2_o and Δt2_v 
            res_o = np.linalg.norm(Δt2_o)
            res_v = np.linalg.norm(Δt2_v)
            res = np.sqrt(res_o**2 + res_v**2)
            #res = res_o + res_v
            print(f'    Initial residual for t2 correction: {res:.3e}')

            Δt2_o_it_save = np.copy(Δt2_o)
            Δt2_v_it_save = np.copy(Δt2_v)

  #          Δt2_o_it = np.copy(Δt2_o)
  #          Δt2_v_it = np.copy(Δt2_v)

            Δt2_o = Δt2_o / eia_o
            Δt2_v = Δt2_v / eia_v

           #set active part to zero
            #Δt2_o[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0
            #Δt2_v[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0

            count = 0 
            acc = np.inf 
            acc_prev = np.inf
            
            # DIIS acceleration parameters
          #  adiis_o = lib.diis.DIIS()
            adiis_v = lib.diis.DIIS()
          #  adiis_o.min_space = 2
           # adiis_o.space = 15
            adiis_v.min_space = 2
            adiis_v.space = 15
            
            diis_start = 3
            
            # Adaptive damping: aggressive early, conservative later
            damp_init = 0.3
            damp_final = 0.8
            
            # Preconditioning: normalize by expected magnitude scales
            scale_o = np.sqrt(np.mean(eia_o**2))
            scale_v = np.sqrt(np.mean(eia_v**2))

            while (acc > tol and count < count_tol):
                Δt2_o_it = Δt2_o_it_save.copy()
                Δt2_v_it = Δt2_v_it_save.copy()

                # Compute residuals with Fock contributions
                Δt2_o_it -= lib.einsum('jk,Ikab -> Ijab', Foo[np.ix_(act_hole, act_hole)], Δt2_o)
                Δt2_o_it -= lib.einsum('IK,Kjab -> Ijab', Foo[np.ix_(inact_hole, inact_hole)], Δt2_o)
                Δt2_o_it /= eia_o
                
                Δt2_v_it += lib.einsum('bc,ijAc -> ijAb', Fvv[np.ix_(act_particle, act_particle)], Δt2_v)
                Δt2_v_it += lib.einsum('AC,ijCb -> ijAb', Fvv[np.ix_(inact_particle, inact_particle)], Δt2_v)
                Δt2_v_it /= eia_v

            #set active part to zero
                #Δt2_o[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0
                #Δt2_v[np.ix_(act_hole, act_hole, act_particle, act_particle)] = 0.0
                


                # Compute convergence criteria before update
                acc_o = np.linalg.norm(Δt2_o_it)
                acc_v = np.linalg.norm(Δt2_v_it)
                acc = np.sqrt(acc_o**2 + acc_v**2)

                # Adaptive damping based on convergence rate
                if count == 0:
                    damp = damp_init
                else:
                    # Progress-based damping: stronger damping when progress stalls
                    rel_improvement = (acc_prev - acc) / (acc_prev + 1e-12)
                    if rel_improvement < 0.05:  # Stalled convergence
                        damp = min(damp_final, damp + 0.05)
                    elif rel_improvement > 0.3:  # Good progress
                        damp = max(damp_init, damp - 0.02)
                    else:
                        damp = damp_init + (damp_final - damp_init) * (count / max(count_tol, 1)) ** 1.5


                damp = damp_init + (damp_final - damp_init) * (count / count_tol) ** 1.5

                # Apply DIIS acceleration if converging well
                if count >= diis_start:
                    try:
                        # Combine amplitude updates for DIIS
                        #Δt2_o_vec = self.run_diis_Δt2(Δt2_o - (1.0 - damp) * Δt2_o_it, adiis_o)
                        Δt2_v_vec = self.run_diis_Δt2(Δt2_v - (1.0 - damp) * Δt2_v_it, adiis_v)
                        
                        Δt2_o = Δt2_o - Δt2_o_it
                        Δt2_v = Δt2_v_vec
                    except:
                        # Fall back to damped update if DIIS fails
                        Δt2_o = Δt2_o - (1.0 - damp) * Δt2_o_it
                        Δt2_v = Δt2_v - (1.0 - damp) * Δt2_v_it
                else:
                    # Standard damped update in early iterations
                    Δt2_o = Δt2_o - (1.0 - damp) * Δt2_o_it
                    Δt2_v = Δt2_v - (1.0 - damp) * Δt2_v_it

                count += 1
                
                # Diagnostic output
                if acc_prev != np.inf:
                    rel_improvement = (acc_prev - acc) / (acc_prev + 1e-12) * 100
                    print(f'    It: {count},  acc occ: {acc_o:.2e},  acc vir: {acc_v:.2e},  '
                          f'total: {acc:.2e},  rel_impr: {rel_improvement:+.1f}%,  damp: {damp:.2f}')
                else:
                    print(f'    It: {count},  acc occ: {acc_o:.2e},  acc vir: {acc_v:.2e},  total: {acc:.2e}')
                
                acc_prev = acc
            
            print(f'    Iter. T2 correction converged in {count}/{count_tol} steps with final accuracy {acc:.2e}')

            Δt2s_o.append(Δt2_o)
            Δt2s_v.append(Δt2_v)

            # Step 11 use t2 active correction to improve Ω

            t2_antisym = 2.0*Δt2_o - np.transpose(Δt2_o, (0, 1, 3, 2))
            Ω[np.ix_(act_particle, inact_hole)] += np.einsum("Ijab,jb -> aI", t2_antisym, Fov[np.ix_(act_hole, act_particle)])
  
            t2_antisym = 2.0*Δt2_v - np.transpose(Δt2_v, (1, 0, 2, 3))
            Ω[np.ix_(inact_particle, act_hole)] += np.einsum("ijAb,jb -> Ai", t2_antisym, Fov[np.ix_(act_hole, act_particle)])

        return Δt2s_o, Δt2s_v , Ω

    def include_t2_active_dense(self, Foo, Fvv, Fov, t2_act, t2_ll, Ω, tol=None, count_tol=None):

        print(f'Computing active t2-correction ...')

        if tol is None:
            tol = self.ll_active_t2_tol
        else:
            tol = min(tol, self.ll_active_t2_tol)
        if count_tol is None:
            count_tol = self.ll_active_t2_max_its

        active_mask = np.zeros(t2_ll.shape, dtype=bool)
        Δt2 = np.zeros_like(t2_ll)

        for k, frag in enumerate(self.frags):
            act_hole = frag[0]
            act_particle = frag[1]
            active = np.ix_(act_hole, act_hole, act_particle, act_particle)
            active_mask[active] = True
            Δt2[active] = t2_act[k] - t2_ll[active]

        def t2_error_residual(delta):
            tmp = lib.einsum("bc,ijac->ijab", Fvv, delta)
            tmp -= lib.einsum("mi,mjab->ijab", Foo, delta)
            return tmp + tmp.transpose(1, 0, 3, 2)

        nonactive = np.flatnonzero((~active_mask).ravel())

        def pack(tensor):
            return tensor.ravel()[nonactive]

        def unpack(vector):
            tensor = np.zeros_like(t2_ll)
            tensor.ravel()[nonactive] = vector
            return tensor

        rhs = -pack(t2_error_residual(Δt2) / self._eris.D)
        rhs_norm = np.linalg.norm(rhs)
        if nonactive.size and rhs_norm > tol:
            operator = scipy.sparse.linalg.LinearOperator(
                (nonactive.size, nonactive.size),
                matvec=lambda vector: pack(
                    t2_error_residual(unpack(vector)) / self._eris.D
                ),
                dtype=t2_ll.dtype,
            )
            residual_history = []
            # Use lgmres (pure Python) instead of gmres: the Fortran-backed
            # gmres overflows its 32-bit workspace-size check once the system
            # size grows large, regardless of the tol/atol kwargs used.
            correction, info = scipy.sparse.linalg.lgmres(
                operator,
                rhs,
#               tol=min(1.0e-8, tol),
                tol=1.0e-8,
                inner_m=min(nonactive.size, 50),
                maxiter=count_tol,
                callback=residual_history.append,
            )
            Δt2.ravel()[nonactive] += correction
            acc = np.linalg.norm(pack(t2_error_residual(Δt2) / self._eris.D))
            if info != 0:
                raise RuntimeError(
                    "Dense active T2 correction did not converge: "
                    f"info={info}, residual={acc:.3e}, target={tol:.3e}"
                )
            print(
                f'    GMRES T2 correction converged in {len(residual_history)} '
                f'iterations with final accuracy {acc:.2e}'
            )
        else:
            acc = rhs_norm
            print(f'    T2 correction already converged with final accuracy {acc:.2e}')

        t2_new = t2_ll + Δt2
        for k, frag in enumerate(self.frags):
            act_hole = frag[0]
            act_particle = frag[1]
            active = np.ix_(act_hole, act_hole, act_particle, act_particle)
            t2_new[active] = t2_act[k]
            Ω[np.ix_(act_particle, act_hole)] = 0.0

        Δt2_antisym = 2.0 * Δt2 - np.transpose(Δt2, (0, 1, 3, 2))
        Ω += lib.einsum("ijab,jb->ai", Δt2_antisym, Fov)

        for frag in self.frags:
            act_hole = frag[0]
            act_particle = frag[1]
            Ω[np.ix_(act_particle, act_hole)] = 0.0

        return t2_new, Ω

        
    def init_amps_fact(self):
       
        Y = self._eris.Lov[:, None, :, :] *self._eris.dD.transpose(2, 0, 1)[None, :, :, :]
        t1 = self._eris.fov/self._eris.eia

        return t1, Y

    def init_amps(self):

        # NOTE Check the initialization here!
        t2 = -lib.einsum("Lia,Ljb->ijab", self._eris.Lov, self._eris.Lov)
        t2 /= self._eris.D     
        t1 = -self._eris.fov/self._eris.eia
        Y = -self._eris.Lov.transpose(0,2,1)[:, None, :, :] *self._eris.dD.transpose(2, 1, 0)[None, :, :, :]
        
        self._Y = Y
        self._t2 = t2
        t1*= 0.0
        t2*= 0.0
        return t1, t2

    def get_t2(self, Y, t2_act, Δt2s_o, Δt2s_v):

        t2 = -lib.einsum("LRai, LRbj -> ijab", Y, Y)
        n_aux, n_rank, n_vir, n_occ = Y.shape
        for k, frag in enumerate(self.frags):
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)

            t2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act[k]

            t2[np.ix_(inact_hole, act_hole, act_particle, act_particle)] += Δt2s_o[k]
            t2[np.ix_(act_hole, act_hole, inact_particle, act_particle)] += Δt2s_v[k]

        return t2

    def get_t2_factorized_laplace(self, Y, t2_act, Δt2s_o=None, Δt2s_v=None):

        t2 = -lib.einsum("LRai, LRbj -> ijab", Y, Y)
        n_aux, n_rank, n_vir, n_occ = Y.shape
        for k, frag in enumerate(self.frags):
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)

            t2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act[k]

            if Δt2s_o is not None and Δt2s_v is not None:

              if Δt2s_o[k].size:
                 t2[np.ix_(inact_hole, act_hole, act_particle, act_particle)] += Δt2s_o[k]
                 t2[np.ix_(act_hole, inact_hole, act_particle, act_particle)] += (
                    Δt2s_o[k].transpose(1, 0, 3, 2)
                 )

              if Δt2s_v[k].size:
                 t2[np.ix_(act_hole, act_hole, inact_particle, act_particle)] += Δt2s_v[k]
                 t2[np.ix_(act_hole, act_hole, act_particle, inact_particle)] += (
                    Δt2s_v[k].transpose(1, 0, 3, 2)
                 )
        return t2

    def get_t2_dense(self, t2_ll, t2_act, Δt2s_o, Δt2s_v):

        t2 = t2_ll.copy()
        n_occ, _, n_vir, _ = t2.shape
        for k, frag in enumerate(self.frags):
            act_hole = frag[0]
            inact_hole = np.delete(range(n_occ), act_hole)
            act_particle = frag[1]
            inact_particle = np.delete(range(n_vir), act_particle)

            t2[np.ix_(act_hole, act_hole, act_particle, act_particle)] = t2_act[k]
            t2[np.ix_(inact_hole, act_hole, act_particle, act_particle)] += Δt2s_o[k]
            t2[np.ix_(act_hole, act_hole, inact_particle, act_particle)] += Δt2s_v[k]

        return t2

    def get_energy(self, t1, t2):                                                     
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

    #Note: For the time being, we will not use the most optmal way to update the amplitudes. It can be taken care later on..
