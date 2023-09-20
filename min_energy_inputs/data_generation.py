# data generation script for minimum energy control inputs

import numpy as np
import matplotlib.pyplot as plt
import math

# function which takes A, B, IS and control horizon, giving inputs and outputs

def state_space(A, B, xi, U, T):
  
  # given xi at time t= 0 and u vectors over control horizon T, return xf at time T
  for i in range(0, T):
     xc = np.dot(A, xi) + np.dot(B, U[:, i])
     xi = xc
  
  return(xi)



def data_gen(A, B, xi, T):

  # given A, B, xf, x0, and T, return a dataset of xi, xf, T and umin.
  p = np.dot(A, A)
  pr = np.dot(A,B)
  CT = np.concatenate((B, pr), axis = 1)

  # calculate input sets for a time horizon T
  u0 = np.random.rand(4,1)
  u1 = np.random.rand(4,1)
  U = np.concatenate((u0, u1), axis = 1)

  for k in range(0, T-2):
      u = np.random.rand(4,1)
      U = np.concatenate((U, u), axis = 1)

  xf = state_space(A, B, xi, U, T)


  for i in range(0, T-2):
    p = np.dot(p, A)

  term1 = xf - np.dot(p, xi)  
  

  for i in range(0, T-1):
    pr = np.dot(A, pr)
    CT = np.concatenate((CT, pr), axis = 1)
  
  term2 = np.linalg.pinv(CT)
  umin = np.dot(term2, term1)
  
  inputs_x = np.concatenate((xi, xf), axis = 0)
  inputs = np.reshape(np.concatenate((inputs_x, [[T]]), axis = 0), (1, 7))
  
  
  output = np.reshape(umin, (1,4))
  
  data = np.concatenate((inputs, output), axis = 0)
  
  return data

  


# function call  

# Useful Equation: x(t+1) = Ax(t) + Bu(t)
# matrices considered: diagonal and coupled (5 each)


filename = 'data.txt'
f = open(filename, 'w')
# for general systems
for i in range(0, 5):
   ac = np.random.rand(3,3)         # random matrix A
   b = np.random.rand(3, 4)         # random matrix B
   
   T = [4,5,6,7,8,9,10]

   for t in T:                      #iterate over all control horizons
      for j in range(0, 100):
        xi = np.random.rand(3,1)    # initial state 
        
        data = data_gen(ac, b, xi, t)
        data.tofile(f, sep=' ', format='%22.14e')
        f.write('\n')
        
        
        

# for decoupled dynamical systems

for i in range(0,5):
  d = np.random.random_integers(1, 100, 3)
  ac = np.diag(d)
  b = np.random.rand(3,4)

  T = [4,5,6,7,8,9,10]

  for t in T:
      for j in range(0, 100):
        xi = np.random.rand(3,1)      # initial state 

        data = data_gen(ac, b, xi, t)
        data.tofile(f, sep=' ', format='%22.14e')
        f.write('\n')
 


  


   