import numpy as np
import scipy.io as sio
import scipy.misc as smisc
from scipy.optimize import fminbound


def to_rgb(img):

    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def to_double(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    return img


def save_mat(img, path):
    sio.savemat(path, {'img':img})


def save_img(img, path):
    img = to_rgb(img)
    smisc.imsave(path, img.round().astype(np.uint8))


def addwgn(x, inputSnr):
    noiseNorm = np.linalg.norm(x.flatten('F')) * 10**(-inputSnr/20)
    xBool = np.isreal(x)
    real = True
    for e in np.nditer(xBool):
        if e == False:
            real = False
    if (real == True):
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1])
    else:
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1]) + 1j * np.random.randn(np.shape(x)[0],np.shape(x)[1])
    
    noise = noise/np.linalg.norm(noise.flatten('F')) * noiseNorm
    y = x + noise
    return y, noise


def optimizeTau(x, algoHandle, taurange, maxfun=10):
    # maxfun ~ number of iterations for optimization
    evaluateSNR = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))
    fun = lambda tau: -evaluateSNR(x,algoHandle(tau)[0])
    tau, fval, _, _ = fminbound(fun, taurange[0],taurange[1], xtol = 1e-6, maxfun = maxfun, disp = 3, full_output = True)
    return tau, fval
