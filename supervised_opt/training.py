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
data_file = os.path.join(curr_dir, "data.txt")
supervised_data = os.path.join(curr_dir, "u_data.txt")

data = np.loadtxt(data_file)
m = np.shape(data)[0]
n = np.shape(data)[1]
xt = data[0:80, 64:88]
yt = data[0:80, 88:91]

xv = data[80:90, 64:88]
yv = data[80:90, 88:91]
print(m)
# print(n)
sdata = np.loadtxt(supervised_data)

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

    model.add(Dense(3, input_shape=(24,), activation = actf, use_bias = False))

    model.add(Dense(3))
    
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])

    return model


class Trainingplot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        # self.test_losses = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        if epoch > 1 and epoch % 50 == 0:  # callback function
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))

            plt.figure(figsize=(10, 6))
            plt.semilogy(N, self.losses, label='Train loss')
            plt.semilogy(N, self.val_losses, label='Validation loss')
            
      
            plt.title('After epoch = {}'.format(epoch))
            plt.xlabel('Epoch #')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('losses.png')



def save_performance(plot_losses, model):
    train_losses = plot_losses.losses
    val_losses = plot_losses.val_losses

    out_dir = 'performance'
    os.makedirs(out_dir, exist_ok = True)

    # np.savetxt(join(out_dir, "u_norm_loss_{}.txt".format(lb)), u)
    # np.savetxt(join(out_dir, "x_norm_loss_{}.txt".format(lb)), x)
    np.savetxt(join(out_dir, "train_loss.txt"), train_losses)
    np.savetxt(join(out_dir, "val_loss.txt"), val_losses)
    model.save_weights(join(out_dir, "weights.h5"))


def train_mlp(model, N):
    plot_losses = Trainingplot()
    model.fit(xt, yt, batch_size = 32, epochs = N, shuffle = True, validation_data = (xv, yv), callbacks = [plot_losses])    
    save_performance(plot_losses, model)



# function calls
para = dict()
para["A"] = 'lrelu'              # activation function for hidden layers
para["v"] = 1.0e-3             # activation function parameter
N = 9000

print(para)

model = NN(para)
train_mlp(model, N)

model.summary()

print('Training loss:', model.evaluate(xt, yt))
print('Validation loss:', model.evaluate(xv, yv))





# # plot ideal and predicted u_min
# U = data[:,0:64]
# print(np.shape(U)[0])
# for i in range(8):
#     j = 8*i
#     a = alpha[:, i]

#     if i == 0:
#        pred_umin = U[:, j:j+8] * a[:, tf.newaxis]
#     else:
#         pred_umin += U[:, j:j+8] * a[:, tf.newaxis]   




# print(np.shape(pred_umin))

# ideal_u = sdata[:,91:99]

# print(pred_umin[0])
# print(ideal_u[0])

# N = np.arange(0, np.shape(U[0:40])[0])

# pred_u_norm = tf.norm(pred_umin, ord = 'euclidean', axis = -1)
# ideal_u_norm = tf.norm(ideal_u, ord = 'euclidean', axis = -1)

# plt.figure(figsize = (10, 6))
# plt.plot(N, pred_u_norm[0:40], 'x', label = 'predicted')
# plt.plot(N, ideal_u_norm[0:40],'o', label = 'ideal')
# plt.xlabel('# datapoint')
# plt.ylabel('minimum energy input norm')
# plt.legend()
# plt.savefig('compare_norm_{}.png'.format(lb))