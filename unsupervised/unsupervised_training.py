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
# print(n)
sdata = np.loadtxt(supervised_data)
plot_sdata = sdata
plot_data = data
# normalize inputs

# for i in range(m):
#     scaling_factor = np.max(np.abs(data[i, :]), axis = -1)
#     if scaling_factor > 1.0:
#         data[i,:] = data[i, :]/scaling_factor
#     else:
#         data[i, :] = data[i, :]    
#     # print(data[i,:])

# for i in range(m):
#     scaling = np.max(np.abs(sdata[i,:]), axis = -1)
#     if scaling > 1.0:
#         sdata[i, :] = sdata[i, :]/scaling

#     else:
#         sdata[i,:] = sdata[i,:]    

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

    actf = 'sigmoid'
    model = Sequential()

    model.add(Dense(60, input_shape=(27,), activation=actf))
    # model.add(Dense(40, activation=actf))
    # model.add(Dense(256, activation='tanh', use_bias=True))
    # model.add(Dense(128, activation='tanh', use_bias = True))
    # model.add(Dense(64, activation='tanh', use_bias = True))
    # model.add(Dense(32, activation='tanh', use_bias = True))
    # model.add(Dense(16, activation='tanh', use_bias = True))
    model.add(Dense(8))

    return model



# loss function definition

def loss_func(input, model):
    with tf.GradientTape() as t1:
        t1.watch(input)
        X = input[:,0:24]
        # U = input[:, 0:64]
        xf = input[:,24:27]
        alpha = model(input)
    # print(alpha)
    # print(tf.shape(alpha))
        for i in range(8):
            j = 3 * i
            # k = 8 * i
            a = alpha[:,i]
            if i == 0:

                x_norm = X[:,j:j+3] * a[:, tf.newaxis]
                # u_norm =  U[:,k:k+8]* a[:, tf.newaxis]
            else:
                x_norm = tf.add(x_norm,(X[:,j:j+3] * a[:, tf.newaxis]))
                # print(x_norm[0])
                # u_norm = tf.add(u_norm , (U[:,k:k+8]* a[:, tf.newaxis]))

        xf_norm = xf -  x_norm
        # print(xf_norm[0])
        # uf_norm = u_norm
        x_norm = tf.square(tf.norm(xf_norm, ord = 'euclidean', axis = -1))
        # print(x_norm)
        # u_norm = (tf.norm(uf_norm, ord = 'euclidean', axis = -1))**2

        x_norm = tf.reduce_mean(x_norm, axis= 0)
        # u_norm = tf.reduce_mean(u_norm, axis = -1)

    return x_norm



# define training procedure and parameters
def train(input, lb, epochs):
    # u_norm_loc = np.zeros(epochs)
    # x_norm_loc = np.zeros(epochs)
    loss_loc = np.zeros(epochs)

    tf.keras.backend.clear_session()

    para = dict()
    para["A"] = 'lrelu'              # activation function for hidden layers
    para["v"] = 1.0e-3               # activation function parameter               

    print(para)
    model = NN(para)
    optimizer = keras.optimizers.Adamax(learning_rate = 0.008)
    

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            x_norm = loss_func(input = input, model= model)
            loss = x_norm
            print(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        # u_norm_loc[epoch] = u_norm.numpy()
        # x_norm_loc[epoch] = x_norm.numpy()
        loss_loc[epoch] = loss.numpy()

    return loss_loc, model


def save_performance(loss, model):
    out_dir = 'performance'
    os.makedirs(out_dir, exist_ok = True)

    # np.savetxt(join(out_dir, "u_norm_loss_{}.txt".format(lb)), u)
    # np.savetxt(join(out_dir, "x_norm_loss_{}.txt".format(lb)), x)
    np.savetxt(join(out_dir, "loss.txt"), loss)
    model.save_weights(join(out_dir, "weights.h5"))


# function calls


epochs = 7000

input = data[:,64:91]
input = tf.convert_to_tensor(input)
input = tf.cast(input, tf.float32)
sinput = sdata
sinput = tf.convert_to_tensor(sinput)
sinput = tf.cast(sinput, tf.float32)

lb = 10
loss, model = train(input, lb, epochs)
save_performance(loss, model)
alpha = model.predict(input)
print(alpha)
# print("u_norm:{}".format(u_norm[-1]))
# print("x_norm:{}".format(x_norm[-1]))
print("loss:{}".format(loss[-1]))



N = np.arange(0, len(loss))
# loss plotting
plt.figure(figsize=(15,10))
plt.xlabel('# epoch')
plt.ylabel('loss')
# plt.semilogy(N, u_norm,label = 'u norm')
# plt.semilogy(N, x_norm, label = 'x norm')
plt.semilogy(N, loss, label = 'Total loss')
plt.legend()

plt.savefig('loss_{}.png'.format(lb))


# plot ideal and predicted u_min
U = plot_data[:,0:64]
print(np.shape(U)[0])
for i in range(8):
    j = 8*i
    a = alpha[:, i]

    if i == 0:
       pred_umin = U[:, j:j+8] * a[:, tf.newaxis]
    else:
        pred_umin += U[:, j:j+8] * a[:, tf.newaxis]   




print(np.shape(pred_umin))

ideal_u = plot_sdata[:,91:99]

print(pred_umin[0])
print(ideal_u[0])

N = np.arange(0, np.shape(U)[0])

pred_u_norm = tf.norm(pred_umin, ord = 'euclidean', axis = -1)
ideal_u_norm = tf.norm(ideal_u, ord = 'euclidean', axis = -1)

plt.figure(figsize = (10, 6))
plt.plot(N, pred_u_norm, 'x', label = 'predicted')
plt.plot(N, ideal_u_norm,'o', label = 'ideal')
plt.xlabel('# datapoint')
plt.ylabel('minimum energy input norm')
plt.legend()
plt.savefig('compare_norm_{}.png'.format(lb))






















        
    
           


        


