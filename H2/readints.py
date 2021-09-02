import numpy as np

tei = np.zeros([2,2,2,2])
with open('h2_sto3g_tei.txt','r') as f:
  for line in f:
    w = line.split()
    tei[ int(w[0]), int(w[1]), int(w[2]), int(w[3]) ] = float(w[4])

for i in range(2):
  for j in range(2):
    for k in range(2):
      for l in range(2):
        print(i,j,k,l,tei[i,j,k,l])
