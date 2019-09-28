#author: Zhiqin Xu 许志钦
#email: xuzhiqin@sjtu.edu.cn
#2019-09-24
# coding: utf-8
'''
Reference: 
1 Training behavior of deep neural network in frequency domain: https://arxiv.org/abs/1807.01251
2 Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks: https://arxiv.org/abs/1901.06523
3 Explicitizing an Implicit Bias of the Frequency Principle in Two-layer Neural Networks: https://arxiv.org/abs/1905.10264
4 Theory of the Frequency Principle for General Deep Neural Networks: https://arxiv.org/abs/1906.09235
'''


#import sys

import matplotlib
matplotlib.use('Agg') 


# In[2]:
import pickle
import time, os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

from BasicFunc import mySaveFig, univAprox
from BasicFunc import my_fft, GetFreq, SelectPeakIndex

# In[3]:
isShowPic=0
Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
pos=[Leftp,Bottomp,Widthp,Heightp]
ComputeStepFFT=40
LowFreqDrawAdd=1e-5 # for plot. plot y+this number, in case of 0 in log-log

### for FFT training with Low to High, or High to Low.
CountPeakStep=0 # always start with 0
CountPeakMaxStep=4000 # the maximum iteration for each freq 

SD=0.0  ### noise standard deviation in sample data, used in the fitting function

# y_name='|x|'
y_name='sigmoid'
y_name='sinx'
# y_name='inv x'
def sigmoid(xx):
    return (1 / (1 + np.exp(-xx)))
def func0(xx,SD):
    y_sin=np.sin(xx)+np.sin(5*xx)
    return y_sin

### discretized the func0 by sin_div
def func_to_approx(xx,sin_div,SD):
    y_sin=func0(xx,SD)
    if sin_div==0:
        return y_sin
    out_y = np.round(y_sin/sin_div)
    out_y2 = out_y * sin_div
    return out_y2

R_variable={}  ### used for saved all parameters and data

### mkdir a folder to save all output
R_variable['iscontinue']=0
if R_variable['iscontinue']:
    FolderName='Errordata/%s/'%('50129')
else:
    BaseDir = 'Errordata/'
    subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1)) 
    FolderName = '%s%s/'%(BaseDir,subFolderName)
    if not os.path.isdir(BaseDir):
        os.mkdir(BaseDir)
    os.mkdir(FolderName)
    os.mkdir('%smodel/'%(FolderName))
R_variable['FolderName']=FolderName 
 

### initialization standard deviation
R_variable['astddev']=0.05 # for weight
R_variable['bstddev']=0.05 # for bias terms2

### the length to discretized the continuous function
R_variable['sin_div']=0

### noise standard deviation in sample data, used in the fitting function
R_variable['SD']=SD

### hidden layer structure
# R_variable['hidden_units']=[20,10]
# R_variable['hidden_units']=[40,20]
# R_variable['hidden_units']=[200,200,200,100]
R_variable['hidden_units']=[1500,1500,500,500]
#R_variable['hidden_units']=[1]

# R_variable['hidden_units']=[800,800,400,400]

R_variable['learning_rate']=4e-5
R_variable['learning_rateDecay']=0
R_variable['rateDecayStep']=2000 

### setup for activation function
R_variable['seed']=0
R_variable['ActFuc']=1  ### 0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate
R_variable['ActFuc_kz']=1  ### integer 
R_variable['train_size']=601;  ### training size
R_variable['batch_size']=int(np.floor(R_variable['train_size'])) ### batch size
R_variable['test_size']=int(201)  ### test size
R_variable['x_start']=-np.pi #math.pi*3 ### start point of input
R_variable['x_end']=np.pi #math.pi*3  ### end point of input

R_variable['isRec_y_test']=0 
R_variable['isFFT']=1 #compute FFT or not   
R_variable['ismovie']=0 # make a training movie

R_variable['tol']=3e-3
R_variable['Total_Step']=600000  ### the training step. Set a big number, if it converges, can manually stop training
R_variable['Record_Step']=4    ### every R_step compute Entropy or other values
R_variable['id']=0         ### index for how many step recorded

### Freq len for recording
R_variable['R_freq_len_test']=int(np.min([800,ComputeStepFFT]))   
R_variable['R_freq_len_train']=int(np.min([800,ComputeStepFFT]))  ### Freq len for recording
# R_variable['R_freq_len_long']=int(np.min([800,R_variable['long_size']/2+1]))    ### Freq len for recording

R_variable['y_name']=y_name     ### the target fitting function
R_variable['FolderName']=FolderName   ### folder for save images
 

# initialization for variables
x_start=R_variable['x_start'] 
x_end=R_variable['x_end'] 
R_variable['test_inputs'] =np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['test_size'],
                                                  endpoint=True),[R_variable['test_size'],1])


R_variable['train_inputs']=np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['train_size'],
                                                  endpoint=True),[R_variable['train_size'],1])
# ###randomly select training set from test set
# indperm = np.random.permutation(test_size)
# ind = indperm[0:Size_in]
# R_variable['train_inputs']=R_variable['test_inputs'][ind]
# ###randomly select training set from test set

R_variable['Fs_train']=R_variable['train_size']/(R_variable['x_end']-R_variable['x_start'])
R_variable['Fs_test']=R_variable['test_size']/(R_variable['x_end']-R_variable['x_start'])

R_variable['Freq_train']=GetFreq(R_variable['x_end']-R_variable['x_start'],R_variable['train_size'])
R_variable['Freq_test']=GetFreq(R_variable['x_end']-R_variable['x_start'],R_variable['test_size'])

R_variable['loss_test']=[]
R_variable['loss_train']=[]
R_variable['fft_fit_test']=[] 
R_variable['fft_true_test']=[]
R_variable['fft_0_train']=[]
R_variable['fft_fit_train']=[]
R_variable['fft_true_train']=[]
R_variable['fft_0_test']=[]
R_variable['y_test_all']=[]


R_variable['ismovie']=1

R_variable['y_train_all']=[]

R_variable['y_test']=[]
R_variable['y_true_test']=[]  
colormaptmp=['tab:orange','tab:purple', 'tab:brown', 'tab:pink', 
                             'tab:gray', 'tab:olive', 'tab:cyan', 'tab:cyan']
sin_div=R_variable['sin_div']
test_inputs=R_variable['test_inputs']
train_inputs=R_variable['train_inputs']

t0=time.time() 




# In[16]:


y_0_test = func0(test_inputs,0)
y_0_train = func0(train_inputs,0)
y_true_test = func_to_approx(test_inputs,R_variable['sin_div'],R_variable['SD'])
y_true_train = func_to_approx(train_inputs,R_variable['sin_div'],R_variable['SD'])

R_variable['y_0_train']=y_0_train
R_variable['y_0_test']=y_0_test
R_variable['y_true_test']= y_true_test
R_variable['y_true_train']=y_true_train

if R_variable['isFFT']:   
    fft_0_test=my_fft(y_0_test,R_variable['R_freq_len_test'])
    R_variable['fft_0_test']=fft_0_test
    fft_0_train=my_fft(y_0_train,R_variable['R_freq_len_train'])
    R_variable['fft_0_train']=fft_0_train
    fft_true_test=my_fft(y_true_test,R_variable['R_freq_len_test'])    
    R_variable['fft_true_test']=fft_true_test
    fft_true_train=my_fft(y_true_train,R_variable['R_freq_len_train'])
    R_variable['fft_true_train']=fft_true_train

### compute a FFT for FFT Training.
fft_y_true_train=np.fft.fft(np.squeeze(y_true_train))
fft_y_true_train_complex=np.complex64(fft_y_true_train)

 
if R_variable['isFFT']:
    Sel_fre_true=np.array(R_variable['fft_true_train'][0:ComputeStepFFT])
    Peak_ind1=SelectPeakIndex(Sel_fre_true)
    Peak_ind=Peak_ind1
    Hand_Add_peak=[1]  # sometimes, first few points are also very important. such as x^2
    Hand_Add_peak=[] 
    Peak_ind=np.concatenate([Peak_ind,Hand_Add_peak],axis=0)
    Peak_ind=np.sort(Peak_ind)
    Peak_len=len(Peak_ind)
    Peak_ind=np.int32(Peak_ind)
    Peak_id=0  # count peak for peak training


     
tf.reset_default_graph() 
with tf.variable_scope('Graph',reuse=tf.AUTO_REUSE) as scope:
        # Our inputs will be a batch of values taken by our functions
        x = tf.placeholder(tf.float32, shape=[None, 1], name="x")
        y_true = tf.placeholder_with_default(input=[[0.0]], shape=[None, 1], name="y")
        
        y,w_Sess,b_Sess,L2w_all_out = univAprox(x, R_variable['hidden_units'],
                                                astddev=R_variable['astddev'],bstddev=R_variable['bstddev'],
                                                ActFuc=R_variable['ActFuc'],seed=R_variable['seed'])
        
        with tf.variable_scope('Loss',reuse=tf.AUTO_REUSE):
            loss=tf.reduce_mean(tf.square(y - y_true))
        
        # We define our train operation using the Adam optimizer
        learning_rate=R_variable['learning_rate']
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = adam.minimize(loss)
        # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


saver = tf.train.Saver() 
sess=tf.Session()
if R_variable['iscontinue']:
    saver.restore(sess, "%smodel/model.ckpt"%(FolderName))
else:
    sess.run(tf.global_variables_initializer())
for i in range(R_variable['Total_Step']):
    
    if (i) % R_variable['Record_Step'] == 0:
        y_test, loss_test_tmp,w_tmp,b_tmp= sess.run([y,loss,w_Sess,b_Sess], 
                                                       feed_dict={x: test_inputs, y_true: y_true_test})
        y_train,loss_train_tmp = sess.run([y,loss],feed_dict={x: train_inputs, y_true: y_true_train})
    
        if i==0:
            y_test_ini=y_test
            R_variable['y_test_ini']=y_test_ini
        R_variable['loss_test'].append(loss_test_tmp)
        R_variable['loss_train'].append(loss_train_tmp)  
        if R_variable['isRec_y_test']:
            R_variable['y_test_all'].append(np.squeeze(y_test))
        
        if R_variable['isFFT']:
            R_variable['fft_fit_test'].append(my_fft(y_test,R_variable['R_freq_len_test']))
            R_variable['fft_fit_train'].append(my_fft(y_train,R_variable['R_freq_len_train'])) 
        if R_variable['ismovie']:
            R_variable['y_train_all'].append(y_train)
        
        
        if loss_train_tmp<R_variable['tol']:
            print('total step:%s; total error:%s'%(i,loss_train_tmp))
            break

    indperm = np.random.permutation(R_variable['train_size'])
    ind = indperm[0:R_variable['batch_size']]
    fft_weight_in=np.ones(R_variable['train_size'])    
    _ = sess.run(train_op, feed_dict={x: train_inputs[ind], y_true: y_true_train[ind]})
        
   
        
    if (i%250==0 and i<2000000):
        print('batch: %d, test loss: %f' % (i + 1, R_variable['loss_test'][-1]))
        print('batch: %d, train loss: %f' % (i + 1, R_variable['loss_train'][-1]))
        R_variable['y_test']=y_test
        R_variable['y_train']=y_train
        t1=time.time()
        print('time cost:%s'%(t1-t0))
        shutil.rmtree('%smodel/'%(FolderName))
        os.mkdir('%smodel/'%(FolderName))
        save_path = saver.save(sess, "%smodel/model.ckpt"%(FolderName))
        with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(R_variable, f, protocol=4)
           
        text_file = open("%s/Output.txt"%(FolderName), "w")
        for para in R_variable:
            if np.size(R_variable[para])>20:
                continue
            text_file.write('%s: %s\n'%(para,R_variable[para]))
        
        text_file.close()
        

        plt.figure() 
        ax=plt.gca()
        plt.title(R_variable['y_name'])
        plt.plot(test_inputs, R_variable['y_true_test'],'r--', label='Test_true')
        plt.plot(test_inputs, y_test,'bo-', label='Test_fit')
        plt.plot(train_inputs, R_variable['y_true_train'],'ko', label='Train_true')
        plt.plot(train_inputs, y_train,'mo', label='Train_fit')
        plt.legend()
        plt.title('step=%s,%s'%(i,R_variable['y_name']))
        fntmp = '%sy_%s'%(R_variable['FolderName'],i)
        mySaveFig(plt,fntmp,ax=ax,iseps=0)
        
        plt.figure()
        ax = plt.gca()
        y1 = R_variable['loss_test']
        y2 = R_variable['loss_train']
        plt.plot(y1,'ro',label='Test')
        plt.plot(y2,'g*',label='Train')
        ax.set_xscale('log')
        ax.set_yscale('log')                
        plt.legend()
        plt.title('step=%s'%(i))
        fntmp = '%s/Loss'%(FolderName)
        mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
        
        if R_variable['isFFT']:
            plt.figure()
            plt.subplot(2,1,1)
            ax=plt.gca()
    
            freq_len=R_variable['R_freq_len_train']
            ind = np.arange(freq_len)
            Freq_train=R_variable['Freq_train'][ind]
            y1 = R_variable['fft_true_train'] / R_variable['train_size']
            y2 = R_variable['fft_fit_train'][-1] / R_variable['train_size']
            y3=R_variable['fft_0_train'] / R_variable['train_size']
            ax.set_xlim([0,Freq_train[-1]])
            ax.set_xlim([0,Freq_train[39]])
            plt.semilogy(Freq_train,y1[ind]+LowFreqDrawAdd,'ro-',label='Trn_true')
            plt.semilogy(Freq_train,y2[ind]+LowFreqDrawAdd,'g*-',label='Trn_fit')
            
            if np.not_equal(sin_div,0):
                plt.semilogy(Freq_train,y3[ind]+LowFreqDrawAdd,'k--',label='y_0')
            plt.legend(ncol=2)
            plt.title('step=%s fft y'%(i))
            plt.subplot(2,1,2)
            ax=plt.gca()
            ax.set_xlim([0,Freq_train[-1]*2])
            ax.set_xlim([0,Freq_train[39]])
            plt.semilogy(Freq_train,y1[ind]+LowFreqDrawAdd,'ro-',label='Trn_true')
            freq_len=R_variable['R_freq_len_test']
            ind = np.arange(freq_len)
            Freq_train=R_variable['Freq_test'][ind] 
            y1 = R_variable['fft_true_test'] / R_variable['test_size']
            y2 = R_variable['fft_fit_test'][-1] / R_variable['test_size']
            plt.semilogy(Freq_train,y1[ind]+LowFreqDrawAdd,'k*-',label='Tst_true')
            plt.semilogy(Freq_train,y2[ind]+LowFreqDrawAdd,'g*-',label='Tst_fit')
            plt.legend(loc=0,ncol=2)
            fntmp = '%sfft'%(FolderName)
            mySaveFig(plt, fntmp,ax=ax,iseps=0) 
        
print("for over")    
R_variable['traintime']=time.time()-t0
print(R_variable['traintime'])
 
#save data

with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(R_variable, f, protocol=4)
   
text_file = open("%s/Output.txt"%(FolderName), "w")
for para in R_variable:
    if np.size(R_variable[para])>20:
#        print(para)
        continue
    text_file.write('%s: %s\n'%(para,R_variable[para]))

text_file.close()



#with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#    R_variable  = pickle.load(f)
#FolderName='' 


test_inputs=R_variable['test_inputs']
plt.figure() 
ax=plt.gca()
plt.plot(test_inputs, R_variable['y_test_ini'],'g-', label='initial')
plt.legend(fontsize=18)
ax.set_xlabel('x',fontsize=18)
ax.set_ylabel('y',fontsize=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
ax.set_position(pos, which='both')
fntmp = '%sy_ini'%(FolderName)
mySaveFig(plt, fntmp,ax=ax)

plt.figure() 
ax=plt.gca()
plt.plot(test_inputs, R_variable['y_test'],'bo-', label='DNN')
plt.plot(test_inputs, R_variable['y_true_test'],'r-', label='true')
plt.legend(fontsize=18,loc=3)
ax.set_xlabel('x',fontsize=18)
ax.set_ylabel('y',fontsize=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
ax.set_position(pos, which='both')
fntmp = '%sy_last_test'%(FolderName)
mySaveFig(plt, fntmp,ax=ax,iseps=0)


plt.figure() 
ax=plt.gca()
plt.plot(test_inputs, R_variable['y_true_test'],'b-', label='true')
plt.legend(fontsize=18)
ax.set_xlabel('x',fontsize=18)
ax.set_ylabel('y',fontsize=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
ax.set_position(pos, which='both')
fntmp = '%sy_true'%(FolderName)
mySaveFig(plt, fntmp,ax=ax,iseps=0)

#plot Loss 
LastIndex=len(R_variable['loss_test'])-2
plt.figure()
ax = plt.gca()
y1 = R_variable['loss_test'][0:LastIndex]
y2 = R_variable['loss_train'][0:LastIndex]
plt.plot(y2,'g-',label='Train')  
plt.plot(y1,'r-',label='Test')
ax.set_yscale('log') 
ax.set_xscale('log')  
plt.legend(fontsize=18)
ax.set_xlabel('step',fontsize=18)
ax.set_ylabel('loss',fontsize=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
ax.set_position(pos, which='both')
fntmp = '%sLoss'%(FolderName)
mySaveFig(plt, fntmp,ax=ax,iseps=0)


if R_variable['isFFT']: 
    UsedPeak=np.arange(len(Peak_ind))
    ComputeStepFFT=40    
    # plot fft with important peak     
    plt.figure()
    ax=plt.gca()
    freq_len=R_variable['R_freq_len_train']
    ind = np.arange(ComputeStepFFT)
    Freq_train=R_variable['Freq_train'][ind]
    y1 = R_variable['fft_true_train']/R_variable['train_size']
    plt.semilogy( y1[0:ComputeStepFFT],'r-',label='Trn_true')
    UsedPeaktmp=np.arange(len(Peak_ind))
    plt.semilogy(Peak_ind[UsedPeak],y1[Peak_ind[UsedPeak]]+1e-6,'ks')
    plt.ylim([1e-5,10])
    ax.set_xlabel('freq index',fontsize=18)
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    ax.set_position(pos, which='both')
    fntmp = '%sffttrain-impPeak'%(FolderName)
    mySaveFig(plt, fntmp,ax=ax,iseps=0)
    

    d_step_fft_true_train_all=np.zeros([ComputeStepFFT,len(R_variable['fft_fit_train'])])
    abs_err_all=np.zeros([ComputeStepFFT,len(R_variable['fft_fit_train'])])
    fft_train_fitAll=np.asarray(np.abs(R_variable['fft_fit_train']))
    for itfes in range(ComputeStepFFT):
        tmp1=fft_train_fitAll[:,itfes]
        tmp2=R_variable['fft_true_train'][itfes]
        d_step_fft_true_train_all[itfes,:] = (np.absolute(tmp1-tmp2))/(1e-5+tmp2)
        abs_err_all[itfes,:] = np.absolute(tmp1-tmp2)

    DrawDis=1
    DrawLastStep=len(R_variable['fft_fit_train'])
    z_min=0.1
    z_max=1  
    plt.figure()
    ax=plt.gca()
    im=plt.pcolor(d_step_fft_true_train_all[Peak_ind[UsedPeak],:DrawLastStep:DrawDis], cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_xlabel('epoch',fontsize=18)
    ax.set_ylabel('freq peak index',fontsize=18)
    ax.set_xscale('log')
    ax.set_xlim([1,DrawLastStep])
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    ax.set_yticks([0,1,2]) 
    ax.set_position(pos, which='both')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax,ticks=[0,0.2,0.4,0.6,0.8,1])
    fntmp = '%sPeakDifferencePcolor-impPeak'%(FolderName)
    mySaveFig(plt, fntmp,ax=ax,iseps=0)

