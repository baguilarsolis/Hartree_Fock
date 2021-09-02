import numpy as np
from numpy.linalg import matrix_power

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
	for i in range(B):
		for j in range(B):
			x += B**-2*(Ptilde[i,j]-P[i,j])**2

	return x**0.5

file_name = 'hf_coord_1.8.xyz'
number_of_atoms, atom_type, atom_coordinates = xyz_reader(file_name)

#Number of electrons
N = 10

#Basis Set Size
B = 6

print('\n')
print("Number of atoms: ", number_of_atoms)
print("Atom Type(s): ", atom_type)
print("Coordinates: ", atom_coordinates)  
print("Number of Electrons: ", N)

# Read in nuclear-nuclear repulsion energy

nuc_rep = np.loadtxt("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/vibrationalE_HF/vibrationalE/hf_1.8_sto3g_nuc.txt")
#print("Nuclear Repulsion Energy: ", nuc_rep,'\n')

# Read in overlap matrix

S = np.loadtxt("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/vibrationalE_HF/vibrationalE/hf_1.8_sto3g_ovl.txt")
#print("Overlap Matrix: ",'\n',S, '\n')

# Read in one electron integral

h_core = np.loadtxt("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/vibrationalE_HF/vibrationalE/hf_1.8_sto3g_oei.txt")
#print("H(core): ",'\n',h_core, '\n')

# Read in two electron integral

tei = np.zeros([6,6,6,6])
with open("/Users/briannaaguilar-solis/python_virt_envs/my_first_venv/vibrationalE_HF/vibrationalE/hf_1.8_sto3g_tei.txt",'r') as f:
  for line in f:
    w = line.split()
    tei[ int(w[0]), int(w[1]), int(w[2]), int(w[3]) ] = float(w[4])

#for i in range(B):
  #for j in range(B):
    #for k in range(B):
      #for l in range(B):
        #print(i,j,k,l,tei[i,j,k,l])


#Diagonalization of S

evalS, U = np.linalg.eig(S)
diagS = np.dot(U.T, np.dot(S,U))
#print("Diagonal S: ", diagS)
#print("U: ", U)

#Transformation Matrix (symmetric orthogonalization)

diagS_minushalf =np.diag(np.diagonal(diagS)**-0.5)
X = np.dot(U, np.dot(diagS_minushalf,U.T))
#print("Transformation Matrix (X)",'\n',X, '\n')

# Check overlap matrix

unitary_check = np.dot(X.T,np.dot(S,X))
#print("Unitary matrix check: ",'\n', unitary_check)

# Inital Guess at P 

P = np.zeros((B,B))
P_previous = np.zeros((B,B))
P_list = []


# Iterative Process

threshold = 100
while threshold > 10**-9:

	# Calculate G matrix

	G = np.zeros((B,B))

	for i in range(B):
		for j in range(B):
			for x in range(B):
				for y in range(B):
					G[i,j] += P[x,y]*(tei[i,j,y,x] - (0.5)*tei[i,x,y,j])

	#print("G matrix: ",'\n',G,'\n')

	# Compute Fock Matrix

	F = h_core+(1*G)
	#print("Fock Matrix: ",'\n',F,'\n')


	# Compute transformed Fock Matrix

	F_prime = np.dot(X.T, np.dot(F,X))
	#print("Transformed Fock Matrix: ",'\n',F_prime,'\n')

	# Diagonalize F_prime

	evalF_prime,C_prime  = np.linalg.eigh(F_prime)
	#print(evalF_prime,C_prime)

	# Order eigenvalues and eigenvectors (precaution for calculating P)

	#idx = evalF_prime.argsort()
	#evalF_prime = evalF_prime[idx]
	C_prime = C_prime[:,:N//2]

	#print("New", '\n')
	#print(evalF_prime,C_prime)
	

	# Compute C matrix

	inverse_X = matrix_power(X,-1)
	C = np.dot(X,C_prime)
	#print("C matrix: ", '\n', C, '\n')

	# Form new density matrix

	for i in range(B):
		for j in range(B):
			P[i,j] = 0
			for a in range(int(N/2)):
				P[i,j] += 2*C[i,a]*C[j,a]
	
	#print("P norm = ", np.linalg.norm(P))
	#print("P trace = ", np.trace(P))

	# Electronic Energy

	E = 0
	for i in range(B):
		for j in range(B):
			E += 0.5*P[j,i]*(h_core[i,j]+F[i,j])

    #print("Energy: ", E)

    #Total Energy
	E_total = E + nuc_rep
	print ("Energy:", E)

	#print("New Density Matrix: ",'\n',P,'\n')
	
	P_list.append(P)
	
	threshold = SD_successive_density_matrix_elements(P_previous,P)
	P_previous = P.copy() 

	#if len(P_list) > 20:
		#break


# Output

print('\n')
print('STO3G Restricted Closed Shell HF algorithm took {} iterations to converge'.format(len(P_list))) 
print('\n')
print('The total electronic energy is {} a.u.'.format(E)) 
print('\n')
print('The total energy, including nuclear repulsion is {} a.u.'.format(E_total))
print('\n')


