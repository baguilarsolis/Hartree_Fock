import numpy as np

# Read in nuclear-nuclear repulsion energy

file = open("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/HF_project/aoints_from_pyscf/h2_sto3g_nuc.txt")
nuc_rep = np.loadtxt(file)
print("Nuclear Repulsion Energy: ", nuc_rep)

# Read in overlap matrix 

file = open("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/HF_project/aoints_from_pyscf/h2_sto3g_ovl.txt")
S = np.loadtxt(file)
print("Overlap Matrix: ", S)

# Read in one electron integral 

file = open("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/HF_project/aoints_from_pyscf/h2_sto3g_oei.txt")
h_core = np.loadtxt(file)
print("One Electron Matrix: ", h_core)

# Read in two electron integral 

file = open("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/HF_project/aoints_from_pyscf/h2_sto3g_tei.txt")

twoe_integral = np.zeros((2,2,2,2))
tei_values = np.genfromtxt(file, usecols=4)

twoe_integral = tei_values.reshape((2,2,2,2))
print("Two-Electron Integral: ",'\n', twoe_integral)


