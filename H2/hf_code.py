import numpy as np

def xyz_reader(file_name):

    # Function reads an xyz file and returns the number of atoms, atom types, and atom coordinates

    file = open(file_name, 'r')

    # Define variables

    number_of_atoms = 0
    atom_type = []
    atom_coordinates = []

    # Loop through file

    for idx, line in enumerate(file):
        # Get number of atoms
        if idx == 0:
            try:
                number_of_atoms = line.split()[0]
            except:
                print(
                    "xyz file not in correct format. Make sure the format follows: https://en.wikipedia.org/wiki/XYZ_file_format")

        # Skip the comment/blank line
        if idx == 1:
            continue

        # Get atom type and positions
        if idx != 0:
            split = line.split()
            atom = split[0]
            coordinates = [float(split[1]), float(split[2]), float(split[3])]

            # Append data
            atom_type.append(atom)
            atom_coordinates.append(coordinates)

    file.close()

    return number_of_atoms, atom_type, atom_coordinates

def SD_successive_density_matrix_elements(Ptilde,P):
	# Function checks the difference betweenn the two most recent guesses of the density matrix (P)

	x = 0
	for i in range(2):
		for j in range(2):
			x += 2**-2*(Ptilde[i,j]-P[i,j])**2

	return x**0.5

file_name = 'H2.xyz'
number_of_atoms, atom_type, atom_coordinates = xyz_reader(file_name)

#Number of electrons

N = 2

print('\n')
print("Number of atoms: ", number_of_atoms)
print("Atom Type(s): ", atom_type)
print("Coordinates: ", atom_coordinates)  
print("Number of Electrons: ", N)

# Read in nuclear-nuclear repulsion energy

file = open("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/HF_project/aoints_from_pyscf/h2_sto3g_nuc.txt")
nuc_rep = np.loadtxt(file)
#print("Nuclear Repulsion Energy: ", nuc_rep,'\n')

# Read in overlap matrix

file = open("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/HF_project/aoints_from_pyscf/h2_sto3g_ovl.txt")
S = np.loadtxt(file)
#print("Overlap Matrix: ",'\n',S, '\n')

# Read in one electron integral

file = open("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/HF_project/aoints_from_pyscf/h2_sto3g_oei.txt")
h_core = np.loadtxt(file)
print("H(core): ",'\n',h_core, '\n')

# Read in two electron integral

file = open("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/HF_project/aoints_from_pyscf/h2_sto3g_tei.txt")
tei_values = np.genfromtxt(file, usecols=4)
twoe_integral = tei_values.reshape((2,2,2,2))
#print("Two-Electron Integral: ",'\n', twoe_integral,'\n')

#Diagonalization of S

evalS, U  = np.linalg.eig(S)
diagS = np.dot(U.T, np.dot(S,U))

#Transformation Matrix (symmetric orthogonalization)

diagS_minushalf =np.diag(np.diagonal(diagS)**-0.5)
X = np.dot(U, np.dot(diagS_minushalf,U.T))
#print("Transformation Matrix (X)",'\n',X, '\n')

# Inital Guess at P 

P = np.zeros((2,2))
P_previous = np.zeros((2,2))
P_list = []

E_0 = []
E_total = 0

# Iterative Process

threshold = 100
while threshold > 10**-4:

	# Calculate G matrix

	G = np.zeros((2,2))

	for i in range(2):
		for j in range(2):
			for x in range(2):
				for y in range(2):
					G[i,j] += P[x,y]*(twoe_integral[i,j,y,x] - (0.5)*twoe_integral[i,x,y,j])

	#print("G matrix: ",'\n',G,'\n')

	# Compute Fock Matrix

	F = h_core + G
	print("Fock Matrix: ",'\n',F,'\n')

	# Compute transformed Fock Matrix

	F_prime = np.dot(X.T, np.dot(F,X))
	#print("Transformed Fock Matrix: ",'\n',F_prime,'\n')

	# Diagonalize F_prime

	evalF_prime,C_prime  = np.linalg.eig(F_prime)
	#print(evalF_prime,C_prime)

	# Order eigenvalues and eigenvectors (precaution for calculating P)

	idx = evalF_prime.argsort()
	evalF_prime = evalF_prime[idx]
	C_prime = C_prime[:,idx]

	#print("New", '\n')
	#print(evalF_prime,C_prime)

	# Compute C matrix

	C = np.dot(X,C_prime)
	#print("C matrix: ", '\n', C, '\n')

	# Energy Calculations

	# Electronic Energy
	for i in range(2):
		for j in range(2):
			E_0[i,j] += P[i,j]*(h_core[i,j] + F(i,j))

        # Total Energy
        #E_total = E_0 + (1/nuc_rep)


	# Form new density matrix

	for i in range(2):
		for j in range(2):
			for a in range(int(N/2)):
				P[i,j] = 2*C[i,a]*C[j,a]

	print("New Density Matrix: ",'\n',P,'\n')
	
	P_list.append(P)
	
	threshold = SD_successive_density_matrix_elements(P_previous,P)
	P_previous = P.copy() 

# Energy Calculations

#E_0 = evalF_prime[0] + evalF_prime[1]
#E_total = E_0 + (1/nuc_rep)

# Output

print('Eigenvalue matrix: ',evalF_prime)

print('\n')
print('STO3G Restricted Closed Shell HF algorithm took {} iterations to converge'.format(len(P_list))) 
print('\n')
print('The orbital energies are {} and {} Hartrees'.format(evalF_prime[0], evalF_prime[1]))
print('\n')
print('The total electronic energy of the ground state is ',E_0) 
print('\n')
print('The total energy, including nuclear repulsion, is ',E_total)
print('\n')
print('The orbital matrix is: ','\n',C)
print('\n')
print('The density/bond order matrix is: ','\n', P)


