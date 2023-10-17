import numpy as np
import os
# import matplotlib.pyplot as plt
from os.path import join

home_dir = os.path.expanduser('~')
main_dir = join(home_dir, "Control-theory")
this_dir = join(main_dir, "unsupervised")
data_dir = join(this_dir, "unsupervised_data.txt")
ideal_data_dir = join(this_dir, "supervised_data.txt")
# test_dir = join(this_dir, "test_data_unsupervised.txt")

def state_data(xi, U, A, B, xf, T):
    N = (np.shape(U)[0])//8
    inputs = U
    for i in range(0,N):
      m = 8*i
      un = U[m:m+8]
      for j in range(0,T):
        k = 2 * j
        u = un[k:k+2]
        xt = np.dot(A, xi) + np.dot(B, u)
        xi = xt

      inputs =  np.concatenate((inputs, xt), axis = 0)
       
    inputs = np.concatenate((inputs, xf), axis = 0)
    inputs = np.reshape(inputs, (1, 58))
    return(inputs)


def final_state(xi, Uf, A, B, T):
   for i in range(T):
      j = 2 * i
      xt = np.dot(A, xi) + np.dot(B, Uf[j:j+2, :])
      xi = xt
   xf = xi  
   return(xf)  
   

def min_energy_inputs(xi, U, Uf, A, B, T):
   p = np.dot(A, A)
   pr = np.dot(A, B)
   CT = np.concatenate((B, pr), axis = 1)

   xf = final_state(xi, Uf, A, B, T)

   for i in range(0, T-2):
      p = np.dot(p, A)

   term1 = xf - np.dot(p, xi)

   for i in range(0, T-2):
      pr = np.dot(A, pr)
      CT = np.concatenate((CT, pr), axis = 1)

   term2 = np.linalg.pinv(CT)

   umin = np.dot(term2, term1)
   umin = np.reshape(umin, (1,8))
   inputs = state_data(xi, U, A, B, xf, T)
   inputs_ideal = np.concatenate((inputs, umin), axis = 1)

   return (inputs_ideal)   

# function call

A = np.random.rand(3,3)
B = np.random.rand(3,2)

N = 5
T = 4
xi = np.zeros((3,1))

for i in range(0,20):

   U = np.random.rand(8*N, 1)

   uf = np.random.rand(8,1)

   xf = final_state(xi, uf, A, B, T)
   input_data = state_data(xi, U, A, B, xf, T)
   ideal_data = min_energy_inputs(xi, U, uf, A, B, T)

   if os.path.isfile(data_dir) == False:
      np.savetxt('unsupervised_data.txt', input_data)

   else:
      with open('unsupervised_data.txt', 'ab') as f:
         np.savetxt(f, input_data)
      
   if os.path.isfile(ideal_data_dir) == False:
      np.savetxt('supervised_data.txt', ideal_data)
                                                               # store ideal minimum energy inputs in different file
   else: 
      with open('supervised_data.txt', 'ab') as f1:
         np.savetxt(f1, ideal_data)



        