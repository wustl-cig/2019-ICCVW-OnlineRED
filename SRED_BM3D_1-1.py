from DataFidelities.CDMClass import CDMClass
from Regularizers.robjects_tf import *
from iterAlgs import *

import scipy.io as sio
import numpy as np
import os

####################################################
####              HYPER-PARAMETERS               ###
####################################################

# set the random seed, please do not comment this line
np.random.seed(128)

# you can change the save path here
save_root = 'Results/BM3D_1-1'
outs_save_root = save_root + '_outs/'

try:
    os.mkdir(outs_save_root)
    print("Directory ", outs_save_root, " created.")
except FileExistsError:
    print("Directory ", outs_save_root, " already exists.")

####################################################
####              DATA PREPARATION               ###
####################################################

# here we load all 10 test images
data_name = 'testImages'
data = sio.loadmat('Data/{}.mat'.format(data_name), squeeze_me=True)
imgs = np.squeeze(data['img'])

# prepare for the data info
sigSize = np.array(imgs[...,0].shape)

# input signal SNR
inputSNR = 25

# number of transmissions
Nt = 1

# generate coded tomography measurement matrix used for all 10 images.
print()
print('Generating simulated light module measurement matrix . . .')
A = CDMClass.genMeas(sigSize, Nt)
print('. . . Done')
print()

# number of iterations
iters = 2000

####################################################
####                LOOP IMAGES                  ###
####################################################

numImgs = imgs.shape[2]
red_output = {}
red_dist   = np.zeros(iters)
red_snr    = np.zeros(iters)
red_time    = np.zeros(iters)

# select which image you want to reconstruct. By default we use the sixth image.
startIndex = 0
endIndex = 5

for i in range(startIndex,endIndex):
    np.random.seed(128)

    # extract truth
    x = imgs[...,i]
    xtrue = x
    sigSize = np.array(x.shape)

    # measure
    Nt = A.shape[0]
    y = np.zeros([Nt, sigSize[0], sigSize[1]], dtype='complex_')
    for j in range(Nt):
        y[j,...] = np.abs(CDMClass.fmult(x, A[j,...]))

    # add white gaussian noise
    y,_ = util.addwgn(np.reshape(y,[Nt,sigSize[0]*sigSize[1]]), inputSNR)
    y   = y.reshape([Nt,sigSize[0],sigSize[1]])

    ####################################################
    ####                   BM3D                     ####
    ####################################################
    batchSize = 1

    #-- Reconstruction --#
    dObj = CDMClass(y, sigSize, A)

    tau_load = sio.loadmat('./OptimPara/optimizedTauForBM3D_iSNR=25_batch=1_Nt=1_Stochastic_iter=2000/fig={}.mat'.format(i))
    tau = tau_load['tau'][0][0]
    rObj = BM3DClass(sigSize, tau, 5)

    print()
    print('####################')
    print('####    SRED    ####')
    print('####################')
    print()
    
    # - To try out direct DnCNN, set useNoise to False.
    # - To save intermediate results, set is_save to True.
    stepSize = 1/(6+2*tau)                 # performance experiment
    saveFull = [False, False, False, False, False, False]
    save = saveFull[i]

    save_root_fig = ''
    if save:
        save_root_fig = save_root + '_' + str(i)
        try:
            os.mkdir(save_root_fig)
            print("Directory ", save_root_fig, " created.")
        except FileExistsError:
            print("Directory ", save_root_fig, " already exists.")

    red_recon, red_out = redEst(dObj, rObj,
                            stochastic=True, batch_size=batchSize,
                            numIter=iters, step=stepSize, accelerate=False, mode='RED', useNoise=True,
                            is_save=save, save_path=save_root_fig, verbose=True, xtrue=xtrue, xinit=None)  # set useNoise to False if you want to try out direct DnCNN
    red_out['recon'] = red_recon

    # save out info
    red_output['img_{}'.format(i)] = red_out
    outs_save_root = save_root + '_outs/'
    info = 'sigma={}_iSNR={}_Nt={}_batch={}_stochastic=True'.format(5, inputSNR, Nt, batchSize)
    sio.savemat(outs_save_root + info + '.mat', red_output)
    red_dist = red_dist + np.array(red_out['dist'])
    red_snr = red_snr + np.array(red_out['snr'])
    red_time = red_time + np.array(red_out['time'])

    util.save_img(red_recon, outs_save_root + '/iter_{}_img={}.tif'.format(iters, i))