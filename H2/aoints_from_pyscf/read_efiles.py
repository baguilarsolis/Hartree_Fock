import numpy as np

# Read in nuclear-nuclear repulsion energy

file = open("h2_sto3g_nuc.txt")

nuc_rep = np.loadtxt(file)

print(nuc_rep)
