import numpy as np
from pyscf import gto
#from pyscf import dft
#from pyscf import scf

def extract_ao_integrals(mol, prefix):
  """Extracts ao integrals from pyscf and writes them to text files.

       mol -- the pyscf mol object (e.g. created by something like gto.M(....) )
    prefix -- A file name prefix used for writing the integrals to files.
              The file names will be:
                prefix_nuc.txt -- the nuclear repulsion energy
                prefix_ovl.txt -- the overlap matrix, created using numpy's savetxt function
                prefix_oei.txt -- the one-electron integral matrix, created using numpy's savetxt function
                prefix_tei.txt -- the two electron integrals written in our own custom format
  """

  with open(prefix + "_nuc.txt", "w") as f:
    f.write("%.18e\n" % mol.energy_nuc())

  np.savetxt(prefix + "_ovl.txt", mol.intor("int1e_ovlp"))

  np.savetxt(prefix + "_oei.txt", mol.intor("int1e_kin") + mol.intor("int1e_nuc"))

  tei = mol.intor("int2e")

  with open(prefix + "_tei.txt", "w") as f:
    for i in range(tei.shape[0]):
      for j in range(tei.shape[1]):
        for k in range(tei.shape[2]):
          for l in range(tei.shape[3]):
            f.write("%4i %4i %4i %4i    %.18e\n" % (i, j, k, l, tei[i,j,k,l]))


mol = gto.M(atom="H 0 0 0; H 1.0 0 0", basis='sto-3g', unit='Angstrom')

extract_ao_integrals(mol, 'h2_sto3g')

