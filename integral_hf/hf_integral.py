import numpy
from functools import reduce
from pyscf import gto, scf, ao2mo
from pyscf.tools import fcidump

# Set up an HeH molecule with the STO-3G basis
mol = gto.M(atom = 'F 0 0 0; H 0 0 1.733', basis='sto-3g')
# Run RHF on molecule
mf = scf.RHF(mol)
# Get things on kernel
mf.kernel()

c = mf.mo_coeff
h_1e = reduce(numpy.dot, (c.T, mf.get_hcore(), c))
eri = ao2mo.kernel(mol, c)

# h_1e = one electron integrals
# eri = two electron integrals
#c.shape = shape of MO coeff
# mol.nelectron = number of electrons
# ms = spin
fcidump.from_integrals('hf_integral.txt', h_1e, eri, c.shape[1], mol.nelectron, ms=0, nuc=mol.energy_nuc())
