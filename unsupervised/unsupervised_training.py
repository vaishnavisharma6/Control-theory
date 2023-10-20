import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
from keras.layers import Dense
from keras import Sequential
from IPython.display import clear_output
from os.path import join


curr_dir = os.getcwd()
data_file = os.path.join(curr_dir, "unsupervised_data.txt")
supervised_data = os.path.join(curr_dir, "supervised_data.txt")

data = np.loadtxt(data_file)
m = np.shape(data)[0]
n = np.shape(data)[1]
print(m)
print(n)
sdata = np.loadtxt(supervised_data)

# define architecture

def NN(para):
    actf = para["A"]
    actp = para["v"]

    if actf == 'lrelu':
        actf = tf.keras.layers.LeakyReLU(actp)

    if actf == 'elu':
        actf = tf.keras.layers.ELU(1.0)


    ki = tf.keras.initializers.he_normal  # kernel_initializer

    model = Sequential()

    model.add(Dense(512, input_shape=(91,), activation='tanh', use_bias=True))
    # model.add(Dense(64, activation='tanh', use_bias=True))
    # model.add(Dense(32, activation='tanh', use_bias=True))
    # model.add(Dense(16, activation='tanh', use_bias = True))
    model.add(Dense(8))

    return model



# loss function definition

def loss_func(input, model):
    with tf.GradientTape() as t1:
        t1.watch(input)
        X = input[:, 64:88]
        U = input[:, 0:64]
        xf = input[:, 88:91]
        alpha = model(input)
    # print(alpha)
    # print(tf.shape(alpha))
        for i in range(8):
            j = 3 * i
            k = 8 * i
            a = alpha[:, i]
            if i == 0:

                x_norm = X[:, j:j+3] * a[:, tf.newaxis]
                u_norm =  U[:,k:k+8]* a[:, tf.newaxis]
            else:
                x_norm = tf.add(x_norm,(X[:, j:j+3] * a[:, tf.newaxis]))
                u_norm = tf.add(u_norm , (U[:,k:k+8]* a[:, tf.newaxis]))

        xf_norm = xf -  x_norm
        uf_norm = u_norm
        x_norm = (tf.norm(xf_norm, ord = 'euclidean', axis = -1))**2
        u_norm = (tf.norm(uf_norm, ord = 'euclidean', axis = -1))**2

        x_norm = tf.reduce_mean(x_norm, axis= -1)
        u_norm = tf.reduce_mean(u_norm, axis = -1)

    return x_norm, u_norm



# define training procedure and parameters
def train(input, lb, epochs):
    u_norm_loc = np.zeros(epochs)
    x_norm_loc = np.zeros(epochs)
    loss_loc = np.zeros(epochs)

    tf.keras.backend.clear_session()

    para = dict()
    para["A"] = 'lrelu'              # activation function for hidden layers
    para["v"] = 1.0e-3               # activation function parameter               

    print(para)
    model = NN(para)
    optimizer = keras.optimizers.Adam()

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            x_norm, u_norm = loss_func(input = input, model= model)
            loss = (u_norm) + (lb * x_norm)
            print(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        u_norm_loc[epoch] = u_norm.numpy()
        x_norm_loc[epoch] = x_norm.numpy()
        loss_loc[epoch] = loss.numpy()

    return u_norm_loc, x_norm_loc, loss_loc, model


def save_performance(u, x, loss, lb):
    out_dir = 'performance'
    os.makedirs(out_dir, exist_ok = True)

    np.savetxt(join(out_dir, "u_norm_loss_{}.txt".format(lb)), u)
    np.savetxt(join(out_dir, "x_norm_loss_{}.txt".format(lb)), x)
    np.savetxt(join(out_dir, "loss_{}.txt".format(lb)), loss)


# function calls


epochs = 1000

input = data
input = tf.convert_to_tensor(input)
input = tf.cast(input, tf.float32)
sinput = sdata
sinput = tf.convert_to_tensor(sinput)
sinput = tf.cast(sinput, tf.float32)

lb = 10
u_norm, x_norm, loss, model = train(input, lb, epochs)
save_performance(u_norm, x_norm, loss, lb)
alpha = model.predict(input)
print("u_norm:{}".format(u_norm[-1]))
print("x_norm:{}".format(x_norm[-1]))
print("loss:{}".format(loss[-1]))



N = np.arange(0, len(u_norm))
# loss plotting
plt.figure(figsize=(15,10))
plt.xlabel('# epoch')
plt.ylabel('u norm')
plt.semilogy(N, u_norm,label = 'u norm')
plt.semilogy(N, x_norm, label = 'x norm')
plt.semilogy(N, loss, label = 'Total loss')
plt.legend()

plt.savefig('loss_{}.png'.format(lb))


# plot ideal and predicted u_min
U = input[:, 0:64]
print(np.shape(U)[0])
for i in range(8):
    j = 8*i
    a = alpha[:, i]
    pred_umin = U[:, j:j+8] * a[:, tf.newaxis]

print(np.shape(pred_umin))

ideal_u = sinput[:, 91:99]

N = np.arange(0, np.shape(U)[0])

plt.figure(figsize = (10, 6))
plt.plot(N, U[:, 0], 'x', label = 'predicted')
plt.plot(N, ideal_u[:, 0],'o', label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('1st element of u vector')
plt.legend()
plt.savefig('compare_0_{}.png'.format(lb))

plt.figure(figsize = (10, 6))
plt.plot(N, U[:, 1], 'x', label = 'predicted')
plt.plot(N, ideal_u[:, 1], 'o',label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('2nd element of u vector')
plt.legend()
plt.savefig('compare_1_{}.png'.format(lb))

plt.figure(figsize = (10, 6))
plt.plot(N, U[:, 2], 'x', label = 'predicted')
plt.plot(N, ideal_u[:, 2], 'o',label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('3rd element of u vector')
plt.legend()
plt.savefig('compare_2_{}.png'.format(lb))

plt.figure(figsize = (10, 6))
plt.plot(N, U[:, 3], 'x', label = 'predicted')
plt.plot(N, ideal_u[:, 3], 'o', label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('4th element of u vector')
plt.legend()
plt.savefig('compare_3_{}.png'.format(lb))

plt.figure(figsize = (10, 6))
plt.plot(N, U[:, 4], 'x', label = 'predicted')
plt.plot(N, ideal_u[:, 4], 'o', label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('5th element of u vector')
plt.legend()
plt.savefig('compare_4_{}.png'.format(lb))

plt.figure(figsize = (10, 6))
plt.plot(N, U[:, 5], 'x', label = 'predicted')
plt.plot(N, ideal_u[:, 5], 'o', label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('6th element of u vector')
plt.legend()
plt.savefig('compare_5_{}.png'.format(lb))

plt.figure(figsize = (10, 6))
plt.plot(N, U[:, 6], 'x', label = 'predicted')
plt.plot(N, ideal_u[:, 6], 'o', label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('7th element of u vector')
plt.legend()
plt.savefig('compare_6_{}.png'.format(lb))

plt.figure(figsize = (10, 6))
plt.plot(N, U[:, 7], 'x', label = 'predicted')
plt.plot(N, ideal_u[:, 7], 'o', label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('8th element of u vector')
plt.legend()
plt.savefig('compare_7_{}.png'.format(lb))


















        
    
           


        


