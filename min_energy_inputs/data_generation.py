# data generation script for minimum energy control inputs
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def state_space(A, B, xi, U, T):
             
  for i in range(0, T):
     xc = np.dot(A, xi) + np.dot(B, np.reshape(U[:, i], (4,1)))                          # return state at time T
     xi = xc
    
  return(xi)



def data_gen(A, B, xi, T):                                    # save data
  p = np.dot(A, A)
  pr = np.dot(A,B)
  CT = np.concatenate((B, pr), axis = 1)

 
  u0 = np.random.rand(4,1)
  u1 = np.random.rand(4,1)                                   # inputs matrix
  U = np.concatenate((u0, u1), axis = 1)
  
  for k in range(0, T-2):
      u = np.random.rand(4,1)
      U = np.concatenate((U, u), axis = 1)
  
  xf = state_space(A, B, xi, U, T)                           # final state

  for i in range(0, T-2):
    p = np.dot(p, A)

  term1 = xf - np.dot(p, xi)  
  

  for i in range(0, T-2):
    pr = np.dot(A, pr)
    CT = np.concatenate((CT, pr), axis = 1)
  
  term2 = np.linalg.pinv(CT)
  
  umin = np.dot(term2, term1)
  
  for i in range(0, T):
      inputs_x = np.concatenate((xi, xf), axis = 0)
      inputs_t = np.concatenate(([[i]], [[T]]), axis = 0)
      inputs = np.concatenate((inputs_x, inputs_t), axis = 0)
      inputs = np.reshape(inputs, (1,8))
      
      j = 4*i
      output = umin[j:j+4, 0]
      output = np.reshape(output, (1,4))

      single_data = np.concatenate((inputs, output), axis = 1)
     
      if i == 0 and T == 4:
         np.savetxt('data.txt', single_data)

      else:
         with open('data.txt', 'ab') as f:
            np.savetxt(f, single_data)

       

# Equation: x(t+1) = Ax(t) + Bu(t)

# for general systems
for i in range(0, 1):
   ac = np.random.rand(3,3)         # matrix A
   b = np.random.rand(3, 4)         # matrix B
   
   T = [4,5,6,7,8,9,10]
   
   for t in T:                      
      for j in range(0, 100):
        xi = np.random.rand(3,1)    # initial state
        data_gen(ac, b, xi, t)
        
        

# for decoupled dynamical systems
# for i in range(0,5):
#   d = np.random.random_integers(1, 100, 3)
#   ac = np.diag(d)
#   b = np.random.rand(3,4)

#   T = [4,5,6,7,8,9,10]

#   for t in T:
#       for j in range(0, 100):
#         xi = np.random.rand(3,1)      # initial state 
#         z = 1
#         data_gen(ac, b, xi, t, z)

 


  


   