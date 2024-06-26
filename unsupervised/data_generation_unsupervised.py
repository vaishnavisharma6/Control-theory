import numpy as np
import os
# import matplotlib.pyplot as plt
from os.path import join

home_dir = os.path.expanduser('~')
main_dir = join(home_dir, "Control-theory")
this_dir = join(main_dir, "unsupervised")
data_dir = join(this_dir, "unsupervised_data.txt")
ideal_data_dir = join(this_dir, "supervised_data.txt")


def state_data(xi, U, A, B, xf, T):
    N = (np.shape(U)[0])//12
    inputs = U
    for i in range(0,N):
      m = 12*i
      un = U[m:m+12]
      for j in range(0,T):
        k = 3 * j
        u = un[k:k+3]
        xt = np.dot(A, xi) + np.dot(B, u)
        xi = xt

      inputs =  np.concatenate((inputs, xt), axis = 0)
       
    inputs = np.concatenate((inputs, xf), axis = 0)
    inputs = np.reshape(inputs, (1, 123))
    return(inputs)



def final_state(xi, Uf, A, B, T):
   for i in range(T):
      j = 3 * i
      xt = np.dot(A, xi) + np.dot(B, Uf[j:j+3, :])
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
   
   r = np.linalg.matrix_rank(CT)
   print(r)
   term2 = np.linalg.pinv(CT)

   umin = np.dot(term2, term1)
   umin = np.reshape(umin, (1,12))
   inputs = state_data(xi, U, A, B, xf, T)
   inputs_ideal = np.concatenate((inputs, umin), axis = 1)

   return (inputs_ideal)   


# function call

A = np.random.rand(3,3)
B = np.random.rand(3,3)

N = 8
T = 4
xi = np.zeros((3,1))
n = 1

uf = np.random.rand(12,1)

xf = final_state(xi, uf, A, B, T)

for i in range(0,1):
   U = np.random.rand(12*N, 1)
   input_data = state_data(xi, U, A, B, xf, T)
   ideal_data = min_energy_inputs(xi, U, uf, A, B, T)
   # input_data = np.reshape(input_data, (1,91))
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


   
        