"""DFRPA with a post-convergence full-ladder energy correction."""

from pyscf import gto, scf, sylvester_cc
from pyscf.mcscf import avas


mol = gto.M(
    atom="O; O 1 1.4; O 2 1.4 1 116.8",
    basis="cc-pvqz",
    verbose=4,
)
mf = scf.RHF(mol).density_fit(auxbasis="cc-pvqz-ri").run(
    conv_tol=1.0e-12
)

myrpa = sylvester_cc.DFRPA(
    mf,
    mo_coeff=mf.mo_coeff,
    ladder_mode="perturbative",
    ll_con_tol=1.0e-6,
    ll_max_its=200,
)
t1, t2 = myrpa.kernel()
e_corr = myrpa.energy(t1, t2)
e_base = e_corr - myrpa.e_ladder_pert

print(f"Ladder-free DFRPA correlation energy: {e_base:.12f} Eh")
print(f"Perturbative ladder correction:       {myrpa.e_ladder_pert:.12f} Eh")
print(f"Corrected DFRPA correlation energy:   {e_corr:.12f} Eh")
print(f"Corrected DFRPA total energy:         {mf.e_tot + e_corr:.12f} Eh")
