import numpy as np
import os
# import matplotlib.pyplot as plt
from os.path import join

home_dir = os.path.expanduser('~')
main_dir = join(home_dir, "Control-theory")
this_dir = join(main_dir, "unsupervised")
data_dir = join(this_dir, "unsupervised_data.txt")
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
   

# function call

A = np.random.rand(3,3)
B = np.random.rand(3,2)

N = 5
T = 4
xi = np.zeros((3,1))

for i in range(0,10):

   U = np.random.rand(8*N, 1)

   uf = np.random.rand(8,1)

   xf = final_state(xi, uf, A, B, T)
   input_data = state_data(xi, U, A, B, xf, T)

   if os.path.isfile(data_dir) == False:
      np.savetxt('unsupervised_data.txt', input_data)

   else:
      with open('unsupervised_data.txt', 'ab') as f:
         np.savetxt(f, input_data)
      




        