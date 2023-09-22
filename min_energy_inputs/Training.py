import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import LeakyReLU
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as k
from keras.callbacks import EarlyStopping
from os.path import join
import io

curr_dir = os.getcwd()
data_file = os.path.join(curr_dir, "data.txt")
test_file = os.path.join(curr_dir, "test_data.txt")

data = np.loadtxt(data_file)
test_data = np.loadtxt(test_file)

np.random.shuffle(data)
np.savetxt("data.txt", data)

np.random.shuffle(test_data)
np.savetxt("test_data.txt", test_data)

d = np.loadtxt("data.txt")
m = d.shape[0]
n = d.shape[1]

mt = int(0.7*m)
mv = m - mt

dt = np.loadtxt("test_data.txt")
m_test = dt.shape[0]
n_test = dt.shape[1]
print(m_test)
print(n_test)

xt = d[0:mt, 0:8]
yt = d[0:mt, 8:12]

xv = d[mt:-1, 0:8]
yv = d[mt:-1, 8:12]

x_test = dt[0:m_test, 0:8]
y_test = dt[0:m_test, 8:12]


class Trainingplot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.test_losses = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        model = self.model

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0)
        self.test_losses.append(test_loss)


    

        if epoch > 1 and epoch % 50 == 0:  # callback function
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))

            plt.figure(figsize=(10, 6))
            plt.semilogy(N, self.losses, label='Train loss')
            plt.semilogy(N, self.val_losses, label='Validation loss')
            plt.semilogy(N, self.test_losses, label = 'Test loss')
      
            plt.title('After epoch = {}'.format(epoch))
            plt.xlabel('Epoch #')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('losses.png')



def NN(para):
    opt = para["opt"]
    lrate = para["n"]
    actf = para["A"]
    actp = para["v"]
    cost = para["C"]
    reg = para["reg"]
    regp = para["b"]

    if opt == 'adam':
        opt = keras.optimizers.legacy.Adam(lrate)

    if actf == 'lReLU':
        actf = tf.keras.layers.LeakyReLU(actp)
    
    if actf == 'elu':
        actf = tf.keras.layers.ELU(1.0)
    
    if reg == 'l2':
        reg = keras.regularizers.l2(regp)
    
    if cost == 'mean_squared_error':
        cost = keras.losses.mean_squared_error

    ki = tf.keras.initializers.he_normal  # kernel_initializer

    model = Sequential()

    model.add(Dense(60, input_shape=(8,), activation=actf,
                    kernel_regularizer=reg, use_bias=True))
    model.add(Dense(40, activation=actf, kernel_regularizer=reg, use_bias=True, kernel_initializer=ki))
    model.add(Dense(20, activation=actf, kernel_regularizer=reg, use_bias=True, kernel_initializer=ki))
    model.add(Dense(4, activation = 'linear'))

    model.compile(optimizer=opt, loss= cost, metrics=['mse'])

    return model


def save_performance(plot_losses, model, Sb):
    train_losses = plot_losses.losses
    val_losses = plot_losses.val_losses
 

    out_dir = "performance"
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(join(out_dir, "train_losses_batch_size_{}.txt".format(Sb)), train_losses)
    np.savetxt(join(out_dir, "val_losses_batch_size_{}.txt".format(Sb)), val_losses)
 

    model.save_weights(join(out_dir,'min_energy_inputs_weights{}.h5'.format(Sb)))


def trainmlp(model, N, Sb, xv):
    plot_losses = Trainingplot()
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 100)
    result = model.fit(xt, yt, batch_size = Sb, epochs = N, verbose = 0, validation_data = (xv, yv),
              shuffle=True, callbacks=[plot_losses])
    pred = model.predict(xv)
    save_performance(plot_losses, model, Sb)


para = dict()
para["opt"] = 'adam'
para["n"] = 1e-3                 # learning rate
para["A"] = 'elu'              # activation function for hidden layers
para["v"] = 1.0e-3               # activation function parameter
para["C"] = 'mean_squared_error' # cost function
para["reg"] = 'l2'               # regularizer
para["b"] = 1.0e-5               # regularization parameter
N = 2000                       # maximum number of epochs
Sb = 526                         # mini batch size
# R = 3                            # Number of restarts

print(para)


model = NN(para)
pred = trainmlp(model, N, Sb, xv)

model.summary()

print("Training loss: ", model.evaluate(xt, yt))
print("Validation loss:", model.evaluate(xv, yv))


print(model.predict(np.array([[9.171582837282914458e-01, 8.558743579327012796e-01, 5.099792324279039946e-01, 2.222633837782698762e+01, 1.061409974041526461e+01, 3.605158571326493444e+01, 0.000000000000000000e+00, 5.000000000000000000e+00]])))
print(model.predict(np.array([[7.093736547848090712e-01, 9.531747089934933248e-01, 5.045033568210737229e-01, 3.229588365018563678e+01, 1.498341170453137394e+01, 5.195573718505915650e+01, 4.000000000000000000e+00, 6.000000000000000000e+00]])))


xtrue = xv[40:90, 0:8]
ytrue = yv[40:90, 0:4]
pred = model.predict(xtrue)
x = len(ytrue)
N = np.arange(x)

plt.figure(figsize = (10,6))
plt.plot(N, ytrue[:, 0], 'o', label = 'true')
plt.plot(N, pred[:,0], 'x', label = 'prediction')
plt.legend()
plt.savefig('comparison_0.png')
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, ytrue[:, 1], 'o', label = 'true')
plt.plot(N, pred[:,1], 'x', label = 'prediction')
plt.legend()
plt.savefig('comparison_1.png')
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, ytrue[:, 2], 'o', label = 'true')
plt.plot(N, pred[:,2], 'x', label = 'prediction')
plt.legend()
plt.savefig('comparison_2.png')
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, ytrue[:, 3], 'o', label = 'true')
plt.plot(N, pred[:,3], 'x', label = 'prediction')
plt.legend()
plt.savefig('comparison_3.png')

plt.close()


xt_true = x_test[0:50, 0:8]
yt_true = y_test[0:50, 0:4]
yt_pred = model.predict(xt_true)
xn = len(yt_pred)
N = np.arange(xn)


plt.figure(figsize = (10,6))
plt.plot(N, yt_true[:, 0], 'o', label = 'true')
plt.plot(N, yt_pred[:,0], 'x', label = 'prediction')
plt.legend()
plt.savefig('test_comparison_0.png')
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, yt_true[:, 1], 'o', label = 'true')
plt.plot(N, yt_pred[:,1], 'x', label = 'prediction')
plt.legend()
plt.savefig('test_comparison_1.png')
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, yt_true[:, 2], 'o', label = 'true')
plt.plot(N, yt_pred[:,2], 'x', label = 'prediction')
plt.legend()
plt.savefig('test_comparison_2.png')
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, yt_true[:, 3], 'o', label = 'true')
plt.plot(N, yt_pred[:,3], 'x', label = 'prediction')
plt.legend()
plt.savefig('test_comparison_3.png')

plt.close()

