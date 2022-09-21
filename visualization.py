import os
import numpy as np
import matplotlib.pyplot as plt


Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88-Leftp
Heightp = 0.9-Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]

def mySaveFig(pltm, fntmp, fp=0, ax=0, isax=0, iseps=0, isShowPic=0):
    if isax == 1:
        pltm.rc('xtick', labelsize=18)
        pltm.rc('ytick', labelsize=18)
        ax.set_position(pos, which='both')
    fnm = '%s.png' % (fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm = '%s.eps' % (fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp != 0:
        fp.savefig("%s.pdf" % (fntmp), bbox_inches='tight')
    if isShowPic == 1:
        pltm.show()
    elif isShowPic == -1:
        return
    else:
        pltm.close()
        
def plot_heat(x, y, y_pred_total, save_path, peak_nums=4):
    y_fft = my_fft(y)
    idx = SelectPeakIndex(y_fft, endpoint=False)
    idx1 = idx[:peak_nums]
    tmp1 = y_fft[idx1]
    abs_err = np.zeros((len(idx1), y_pred_total.shape[0]))
    
    for i in range(y_pred_total.shape[0]):
        tmp2 = my_fft(y_pred_total[i])[idx1]
        abs_err[:, i] = np.abs(tmp1 - tmp2) / (1e-5 + tmp1)
    
    plt.figure()
    plt.pcolor(abs_err, cmap='RdBu', vmin=0.1, vmax=1)
    plt.colorbar()
    plt.savefig(os.path.join(save_path, 'hot.png'))
        
def plot_loss(loss, save_path):
    fig = plt.figure()
    # Create plot inside the figure
    ax = fig.add_subplot()
    ax.set_xlabel('Epoch #', fontsize=18)
    ax.set_ylabel('loss', fontsize=18)
    # Plot
    epochs = np.arange(0, len(loss))
    ax.plot(epochs, loss, "b-", label='train')
    # ax.plot(epochs, R['test_loss'], "r--",label='test')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    plt.legend(fontsize=18)
    plt.tight_layout()
    fig.canvas.draw()
    fntmp = os.path.join(save_path, 'loss')
    mySaveFig(plt, fntmp, ax=ax, iseps=0)
    plt.close()
    

def plot_info(epoch, x, y, y_pred, save_path, mode='train'):
    fig = plt.figure()
    # Create plot inside the figure
    ax = fig.add_subplot()
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    # Plot

    ax.plot(x, y, "b-", label='true')
    ax.plot(x, y_pred, "r-", label='train')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.legend(fontsize=18)
    plt.tight_layout()
    fig.canvas.draw()
    fntmp = os.path.join(save_path, f'E{epoch}_{mode}.png')
    mySaveFig(plt, fntmp, ax=ax, iseps=0)
    plt.close()

    plt.figure()
    y_fft = my_fft(y)/x.shape[0]
    plt.semilogy(y_fft+1e-5, label='real')
    idx = SelectPeakIndex(y_fft, endpoint=False)
    plt.semilogy(idx, y_fft[idx]+1e-5, 'o')
    y_fft_pred = my_fft(y_pred)/x.shape[0]
    plt.semilogy(y_fft_pred+1e-5, label='train')
    plt.semilogy(idx, y_fft_pred[idx]+1e-5, 'o')
    plt.legend()
    plt.xlabel('freq idx')
    plt.ylabel('freq')
    plt.savefig(os.path.join(save_path, f'fft_{mode}.png'))
    plt.close()

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