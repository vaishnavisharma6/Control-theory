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
m = np.shape(data)
print(m)

# n = np.shape(data)[1]
print(m)
# print(n)
sdata = np.loadtxt(supervised_data)
plot_sdata = sdata
plot_data = data

# define architecture

def NN(para):
    actf = para["A"]
    actp = para["v"]

    if actf == 'lrelu':
        actf = tf.keras.layers.LeakyReLU(actp)

    if actf == 'elu':
        actf = tf.keras.layers.ELU(1.0)

    kr = tf.keras.regularizers.L2

    ki = tf.keras.initializers.he_normal  # kernel_initializer

    # actf = 'sigmoid'
    model = Sequential()

    model.add(Dense(10, input_shape=(120,), activation='sigmoid'))
    model.add(Dense(10, activation = 'sigmoid'))

    model.add(Dense(8))

    return model



# loss function definition

def loss_func(input, model, xf):
    with tf.GradientTape() as t1:
        t1.watch(input)
        X = input[:, 96:120]
        U = input[:, 0:96]
        # xf = input[:,120:123]
        alpha = model(input)
        print(alpha)
    # print(tf.shape(alpha))
        for i in range(8):
            j = 3 * i
            k = 12 * i
            a = alpha[:,i]
            if i == 0:

                x_norm = X[:,j:j+3] * a[:, tf.newaxis]
                u_norm =  U[:,k:k+12]* a[:, tf.newaxis]

            else:
                x_norm = tf.add(x_norm,(X[:,j:j+3] * a[:, tf.newaxis]))
                # print(x_norm[0])
                u_norm = tf.add(u_norm , (U[:,k:k+12]* a[:, tf.newaxis]))

        xf_norm = xf -  x_norm
        # print(xf_norm[0])
        uf_norm = u_norm
        x_norm = tf.square(tf.norm(xf_norm, ord = 'euclidean', axis = -1))
        # print(x_norm)
        u_norm = tf.square(tf.norm(uf_norm, ord = 'euclidean', axis = -1))

        x_norm = tf.reduce_mean(x_norm, axis= 0)
        u_norm = tf.reduce_mean(u_norm, axis = -1)

    return x_norm, u_norm



# define training procedure and parameters
def train(input, xf, lb, epochs):
    u_norm_loc = np.zeros(epochs)
    x_norm_loc = np.zeros(epochs)
    loss_loc = np.zeros(epochs)

    tf.keras.backend.clear_session()

    para = dict()
    para["A"] = 'lrelu'              # activation function for hidden layers
    para["v"] = 1.0e-1           # activation function parameter               

    print(para)
    model = NN(para)
    optimizer = keras.optimizers.Adamax()
    

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            x_norm, u_norm = loss_func(input = input, model= model, xf = xf)
            loss =   (u_norm) + (100 * x_norm)
            print(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        u_norm_loc[epoch] = u_norm.numpy()
        x_norm_loc[epoch] = x_norm.numpy()
        loss_loc[epoch] = loss.numpy()

    return  u_norm_loc, x_norm_loc, loss_loc, model


def save_performance(x_norm, u_norm, loss, model):
    out_dir = 'performance'
    os.makedirs(out_dir, exist_ok = True)

    np.savetxt(join(out_dir, "u_norm.txt"), u_norm)
    np.savetxt(join(out_dir, "x_norm.txt"), x_norm)
    np.savetxt(join(out_dir, "loss.txt"), loss)
    model.save_weights(join(out_dir, "weights.h5"))


# function calls


epochs = 10000
input = data[0:120]
xf = data[120:123]
input = tf.convert_to_tensor(input)
input = tf.cast(input, tf.float32)
input = tf.expand_dims(input, axis = 0)
sinput = sdata
sinput = tf.convert_to_tensor(sinput)
sinput = tf.cast(sinput, tf.float32)
sinput = tf.expand_dims(sinput, axis = 0)

lb = 0.1
u_norm, x_norm, loss, model = train(input, xf, lb, epochs)
save_performance(x_norm, u_norm, loss, model)
alpha = model.predict(input)
print(alpha)
print("u_norm:{}".format(u_norm[-1]))
print("x_norm:{}".format(x_norm[-1]))
print("loss:{}".format(loss[-1]))



N = np.arange(0, len(x_norm))
# loss plotting
plt.figure(figsize=(15,10))
plt.xlabel('# epoch')
plt.ylabel('loss')
plt.semilogy(N, u_norm,label = 'u norm')
plt.semilogy(N, x_norm, label = 'x norm')
plt.semilogy(N, loss, label = 'Total loss')
plt.legend()

plt.savefig('loss.png')


# plot ideal and predicted u_min
U = input[:, 0:96]
print(np.shape(U)[0])
for i in range(8):
    j = 12*i
    a = alpha[:, i]

    if i == 0:
       pred_umin = U[:,j:j+12] * a[:, tf.newaxis]
    else:
        pred_umin = tf.add(pred_umin, U[:,j:j+12] * a[:, tf.newaxis])   




print(np.shape(pred_umin))

ideal_u = sinput[:,123:135]

print(pred_umin)
print(ideal_u)

N = np.arange(0, np.shape(U[0:40])[0])

pred_u_norm = tf.norm(pred_umin, ord = 'euclidean', axis = -1)
ideal_u_norm = tf.norm(ideal_u, ord = 'euclidean', axis = -1)

plt.figure(figsize = (10, 6))
# plt.plot(N, pred_u_norm[0:40], 'x', label = 'prediction')
plt.plot(N, np.abs(ideal_u_norm-pred_u_norm),'o', label = 'difference')
plt.xlabel('# datapoint')
plt.ylabel('minimum energy input norm difference')
plt.legend()
plt.savefig('compare_diff.png')


plt.figure(figsize = (10, 6))
plt.plot(N, pred_u_norm, 'x', label = 'prediction')
plt.plot(N, ideal_u_norm,'o', label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('minimum energy input norm difference')
plt.legend()
plt.savefig('compare_norm.png')



















        
    
           


        


