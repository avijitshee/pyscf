"""DFRPA with the AVAS active-particle Jvv ladder approximation."""

import numpy as np

from pyscf import gto, scf, sylvester_cc
from pyscf.mcscf import avas


mol = gto.M(
    atom="O; O 1 1.4; O 2 1.4 1 116.8",
    basis="cc-pvqz",
)
mol.verbose = 4

# The AVAS step rotates the canonical occupied and virtual RHF subspaces.
mf = scf.RHF(mol).density_fit(auxbasis="cc-pvqz-ri").run(conv_tol=1.0e-12)

openshell_option = 3

ao_labels = ["O 2p", "O 2s", "O 3p", "O 3s", "O 3d"]
minao = "ccpvdz"

avas_obj = avas.AVAS(mf, ao_labels, minao=minao, openshell_option=openshell_option)
avas_obj.with_iao = True
avas_obj.threshold = 1e-7

_, _, mocas = avas_obj.kernel()

act_hole = (
    np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
)

act_particle = np.where(
    avas_obj.vir_weights > avas_obj.threshold
)[0]

myrpa = sylvester_cc.DFRPA(
    mf,
    mo_coeff=mocas,
    act_hole=act_hole,
    act_particle=act_particle,
    ladder_mode="active_jvv",
    ll_con_tol=1.0e-6,
    ll_max_its=200,
)
t1, t2 = myrpa.kernel()
e_corr = myrpa.energy(t1, t2)

print(f"Active-Jvv DFRPA correlation energy: {e_corr:.12f} Eh")
print(f"Active-Jvv DFRPA total energy:       {mf.e_tot + e_corr:.12f} Eh")
