"""Frozen-core dense and Laplace--Sylvester DFRPA comparison.

The ``frozen`` argument follows the restricted CCSD convention.  An integer
freezes that many lowest molecular orbitals; an explicit sequence can instead
select individual occupied or virtual MO indices.
"""

from pathlib import Path

import numpy as np

from pyscf import cc, gto, scf, sylvester_cc


mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="cc-pvqz",
    verbose=4,
)
mf = scf.RHF(mol).density_fit(auxbasis="cc-pvqz-ri").run(
    conv_tol=1.0e-12
)

# Identical to cc.CCSD(mf, frozen=1): remove the lowest-energy oxygen core MO.
frozen = 1
convergence = 1.0e-8
laplace_grid_size = 12
laplace_root = (
    Path(__file__).resolve().parents[2] / "external" / "laplace-minimax"
)

dense = sylvester_cc.DFRPA(
    mf,
    frozen=frozen,
    ladder_mode=None,
    ll_con_tol=convergence,
    ll_max_its=100,
)
t1_dense, t2_dense = dense.kernel()
e_dense = dense.energy(t1_dense, t2_dense)

sylvester = sylvester_cc.DFRPASylvester(
    mf,
    frozen=frozen,
    ladder_mode=None,
    ll_con_tol=convergence,
    ll_max_its=100,
    ll_laplace_root=laplace_root,
    ll_laplace_npoints=laplace_grid_size,
    ll_chebyshev_tol=1.0e-14,
)
t1_sylvester, t2_sylvester = sylvester.kernel()

ccsd_mask = cc.CCSD(mf, frozen=frozen).get_frozen_mask()
energy_difference = sylvester.e_corr - e_dense

print(f"Frozen orbitals:                     {frozen}")
print(f"Active occupied orbitals:            {dense.nocc}")
print(f"Active virtual orbitals:             {dense.nvir}")
print(f"Dense frozen-core correlation:       {e_dense:.12f} Eh")
print(f"Sylvester frozen-core correlation:   {sylvester.e_corr:.12f} Eh")
print(f"Sylvester - dense:                   {energy_difference:.6e} Eh")
print(f"Laplace grid size:                   {laplace_grid_size}")

assert dense.converged
assert sylvester.converged
np.testing.assert_array_equal(dense.get_frozen_mask(), ccsd_mask)
np.testing.assert_array_equal(sylvester.get_frozen_mask(), ccsd_mask)
assert t1_dense.shape == t1_sylvester.shape == (dense.nocc, dense.nvir)
assert t2_dense.shape == t2_sylvester.shape == (
    dense.nocc,
    dense.nocc,
    dense.nvir,
    dense.nvir,
)
assert abs(energy_difference) < 1.0e-7
