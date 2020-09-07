# library
import os
import numpy as np
import time
# scripts
import util


######## Iterative Methods #######

def redEst(dObj, rObj, 
            stochastic=True, batch_size=5, # simpliest case, batch size = 1 
            numIter=100, step=1, accelerate=False, mode='RED', useNoise=True, 
            verbose=False, is_save=False, save_path='result', xtrue=None, xinit=None):
    """
    Regularization by Denoising (RED)
    
    ### INPUT:
    dObj       ~ data fidelity term, measurement/forward model
    rObj       ~ regularizer term
    numIter    ~ total number of iterations
    accelerate ~ use APGM or PGM
    mode       ~ RED update or PROX update
    useNoise.  ~ true if CNN predict noise; false if CNN predict clean image
    step       ~ step-size
    verbose    ~ if true print info of each iteration
    is_save    ~ if true save the reconstruction of each iteration
    save_path  ~ the save path for is_save
    xtrue      ~ the ground truth of the image, for tracking purpose
    xinit      ~ initialization of x (zero otherwise)

    ### OUTPUT:
    x     ~ reconstruction of the algorithm
    outs  ~ detailed information including cost, snr, step-size and time of each iteration

    """
    
    ########### HELPER FUNCTION ###########

    evaluateSnr = lambda xtrue, x: 20*np.log10(np.linalg.norm(xtrue.flatten('F'))/np.linalg.norm(xtrue.flatten('F')-x.flatten('F')))

    ########### INITIALIZATION ###########
    
    # initialize save foler

    if is_save:
        abs_save_path = os.path.abspath(save_path)

    # initialize info data
    if xtrue is not None:
        xtrueSet = True
        snr = []
    else:
        xtrueSet = False

    dist = []
    timer = []
    
    # initialize variables
    if xinit is not None:
        pass
    else:    
        xinit = 1e-2 * np.ones(dObj.sigSize, dtype=np.float32) 
    # outs = struct(xtrueSet)
    x = xinit
    s = x            # gradient update
    t = 1.           # controls acceleration
    p = rObj.init()  # dual variable for TV
    pfull = rObj.init()  # dual variable for TV
    

    ############ RED (EPOCH) ############

    for indIter in range(numIter):

        timeStart = time.time()
        # get gradient
        if stochastic:
            sub = np.random.permutation(dObj.Nt)
            meas_list = list(sub[0:batch_size])
            g = dObj.gradStoc(s, meas_list)  # circle trough 1-by-1
        else:
            g = dObj.grad(s)

        if mode == 'RED':
            g_robj, p = rObj.red(s, step, p, useNoise=useNoise, extend_p=None)
            xnext = s - step*(g + g_robj)
        else:
            print("No such mode option")
            exit()

        timeEnd = time.time() - timeStart

        ########### LOG INFO ###########

        # calculate full gradient for convergence plot
        gfull = dObj.grad(x)

        if mode == 'RED':
            g_robj, pfull = rObj.red(x, step, pfull, useNoise=useNoise, extend_p=None)
            Px = x - step*(g + g_robj)
            # Sx
            diff = np.linalg.norm(g.flatten('F') + g_robj.flatten('F')) ** 2
        else:
            print("No such mode option")
            exit()

        # acceleration
        if accelerate:
            tnext = 0.5*(1+np.sqrt(1+4*t*t))
        else:
            tnext = 1
        s = xnext + ((t-1)/tnext)*(xnext-x)
        
        # output info
        # cost[indIter] = data
        dist.append(diff)
        timer.append(timeEnd)
        # evaluateTol(x, xnext)
        if xtrueSet:
            snr.append(evaluateSnr(xtrue, x))

        # update
        t = tnext
        x = xnext

        # save & print
        if is_save:
            util.save_mat(xnext, abs_save_path+'/iter_{}_mat.mat'.format(indIter+1))
            util.save_img(xnext, abs_save_path+'/iter_{}_img.tif'.format(indIter+1))
        
        if verbose:
            if xtrueSet:
                print('[redEst: '+str(indIter+1)+'/'+str(numIter)+']'+' [||Gx_k||^2/||Gx_0||^2: %.5e]'%(dist[indIter]/dist[0])+' [snr: %.2f]'%(snr[indIter]))
            else:
                print('[redEst: '+str(indIter+1)+'/'+str(numIter)+']'+' [||Gx_k||^2/||Gx_0||^2: %.5e]'%(dist[indIter]/dist[0]))

        # summarize outs
        outs = {
            'dist': dist/dist[0],
            'snr': snr,
            'time': timer
        }

    return x, outs