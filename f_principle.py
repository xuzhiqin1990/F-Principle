import os
import logging
import ml_collections

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


os.environ["CUDA_VISIBLE_DEVICES"] = '5'


# basic function to approx
def func0(xx):
    y_sin = np.sin(xx)+2*np.sin(3*xx)+3*np.sin(5*xx)
    return y_sin


def func_to_approx(xx, alpha):
    y_sin = func0(xx)
    if alpha == 0:
        return y_sin
    out_y = np.round(y_sin/alpha)
    out_y2 = out_y * alpha
    return out_y2


# get train data
def get_data(config):
    train_input = np.reshape(np.linspace(config.x_start, config.x_end,
                             num=config.train_size, endpoint=True), [config.train_size, 1]).astype(np.float32)

    y_train = func_to_approx(train_input, config.alpha)

    return (train_input, y_train)


#
def my_fft(data, freq_len=40, x_input=np.zeros(10), kk=0, min_f=0, max_f=np.pi/3, isnorm=1):
    second_diff_input = np.mean(np.diff(np.diff(np.squeeze(x_input))))
    if abs(second_diff_input) < 1e-10:
        datat = np.squeeze(data)
        datat_fft = np.fft.fft(datat)
        ind2 = range(freq_len)
        fft_coe = datat_fft[ind2]
        if isnorm == 1:
            return_fft = np.absolute(fft_coe)
        else:
            return_fft = fft_coe
    else:
        return_fft = get_ft_multi(
            x_input, data, kk=kk, freq_len=freq_len, min_f=min_f, max_f=max_f, isnorm=isnorm)
    return return_fft


def get_ft_multi(x_input, data, kk=0, freq_len=100, min_f=0, max_f=np.pi/3, isnorm=1):
    n = x_input.shape[1]
    if np.max(abs(kk)) == 0:
        k = np.linspace(min_f, max_f, num=freq_len, endpoint=True)
        kk = np.matmul(np.ones([n, 1]), np.reshape(k, [1, -1]))
    tmp = np.matmul(np.transpose(data), np.exp(-1J * (np.matmul(x_input, kk))))
    if isnorm == 1:
        return_fft = np.absolute(tmp)
    else:
        return_fft = tmp
    return np.squeeze(return_fft)


def SelectPeakIndex(FFT_Data, endpoint=True):
    D1 = FFT_Data[1:-1]-FFT_Data[0:-2]
    D2 = FFT_Data[1:-1]-FFT_Data[2:]
    D3 = np.logical_and(D1 > 0, D2 > 0)
    tmp = np.where(D3 == True)
    sel_ind = tmp[0]+1
    if endpoint:
        if FFT_Data[0]-FFT_Data[1] > 0:
            sel_ind = np.concatenate([[0], sel_ind])
        if FFT_Data[-1]-FFT_Data[-2] > 0:
            Last_ind = len(FFT_Data)-1
            sel_ind = np.concatenate([sel_ind, [Last_ind]])
    return sel_ind


# def metrics we need to use in test_step
class MyModel(tf.keras.Model):
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        # self.compiled_loss(y, y_pred)
        # self.compiled_metrics.update_state(y, y_pred)

        return_metrics = {}
        # for metric in self.metrics:
        # return_metrics[metric.name] = metric.result()
        return_metrics['y_pred'] = y_pred

        return return_metrics


# def what to do in the end of every epoch
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info('Epoch: {}/{}, train loss: {:.6f}'.format(
            epoch, self.params['epochs'], logs['loss']))


def main():
    config_dict = {
        "alpha": 0,
        "x_start": -5,
        "x_end": 5,
        "train_size": 101,
        "layer_list": [200, 200, 200, 100],
        "ActFun": "tanh",
        "dims_input": 1,
        "dims_output": 1,
        "lossfunc": "MSE",
        "optimizer": "Adam",
        "epochs": 3000,
        "batch_size": 50,
        "lr": 2e-4,
        "result_dir": 'results/f_principle'
    }
    config = ml_collections.ConfigDict(config_dict)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    input = tf.keras.Input(shape=(config.dims_input, ))
    x = input
    for layer in config.layer_list:
        x = tf.keras.layers.Dense(layer, activation=config.ActFun)(x)
    output = tf.keras.layers.Dense(config.dims_output)(x)

    model = MyModel(input, output)

    train_data = get_data(config)
    x, y = train_data
    opt = tf.keras.optimizers.__dict__[config.optimizer](lr=config.lr)
    model.compile(optimizer=opt, loss='mse')
    history = model.fit(x, y, batch_size=101,
                        epochs=config.epochs,
                        validation_data=train_data,
                        callbacks=[CustomCallback()],
                        verbose=0)

    # plot loss function
    train_loss = history.history['loss']
    plt.figure()
    plt.semilogy(train_loss, label='train')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(config.result_dir + '/loss.png')
    plt.figure()

    # plot the convergence of the peak idx of frequence
    y_pred = model.predict(x)
    y_fft = my_fft(y)/config.train_size
    plt.semilogy(y_fft+1e-5, label='real')
    idx = SelectPeakIndex(y_fft, endpoint=False)
    plt.semilogy(idx, y_fft[idx]+1e-5, 'o')
    y_fft_pred = my_fft(y_pred)/config.train_size
    plt.semilogy(y_fft_pred+1e-5, label='train')
    plt.semilogy(idx, y_fft_pred[idx]+1e-5, 'o')
    plt.legend()
    plt.xlabel('freq idx')
    plt.ylabel('freq')
    plt.savefig(config.result_dir + '/fft.png')

    y_pred_epoch = np.squeeze(history.history['val_y_pred'])
    idx1 = idx[:4]
    abs_err = np.zeros([len(idx1), config.epochs])
    y_fft = my_fft(y)
    tmp1 = y_fft[idx1]
    for i in range(len(y_pred_epoch)):
        tmp2 = my_fft(y_pred_epoch[i])[idx1]
        abs_err[:, i] = np.abs(tmp1 - tmp2)/(1e-5 + tmp1)

    plt.figure()
    plt.pcolor(abs_err, cmap='RdBu', vmin=0.1, vmax=1)
    plt.colorbar()
    plt.savefig(config.result_dir + '/hot.png')


if __name__ == '__main__':
    main()
