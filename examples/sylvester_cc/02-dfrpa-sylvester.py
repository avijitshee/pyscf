"""Laplace--Sylvester DFRPA calculation for canonical ozone orbitals."""

from pathlib import Path

from pyscf import gto, scf, sylvester_cc


mol = gto.M(
    atom="O; O 1 1.6; O 2 1.6 1 116.8",
    basis="cc-pvtz",
)

mol.verbose = 4

# DFRPASylvester requires the canonical MOs from the RHF calculation.
mf = scf.RHF(mol).density_fit(auxbasis="cc-pvtz-ri").run(conv_tol=1.0e-12)

laplace_root = (
    Path(__file__).resolve().parents[2] / "external" / "laplace-minimax"
)
laplace_grid_size = 16

myrpa = sylvester_cc.DFRPASylvester(
    mf,
    ll_con_tol=1.0e-8,
    ll_max_its=100,
    ll_laplace_root=laplace_root,
    ll_laplace_npoints=laplace_grid_size,
)
t1, t2 = myrpa.kernel()

print(f"DFRPASylvester correlation energy: {myrpa.e_corr:.12f} Eh")
print(f"DFRPASylvester total energy:       {myrpa.e_tot:.12f} Eh")
print(f"Laplace grid size:                 {laplace_grid_size}")
print(f"Converged:                         {myrpa.converged}")

assert myrpa.converged
