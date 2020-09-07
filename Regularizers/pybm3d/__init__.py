#############################################################################
#
#  BM3D is an algorithm for attenuation of additive spatially correlated
#  stationary (aka colored) Gaussian noise in grayscale images.
#
#  K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Image restoration by
#  sparse 3D transform-domain collaborative filtering," Proc. SPIE Electronic
#  Imaging '08, vol. 6812, no. 681207, San Jose (CA), USA, January 2008.
#  DOI: http://doi.org/10.1117/12.766355
#
#  FUNCTION INTERFACE:
#
#  y_est = BM3D(z, sigmaPSD, profile)
#
#          'z'  noise image
#
#  INPUT ARGUMENTS:
#
#         'z' : noisy image (M x N double array, intensities in range [0,1])
#  'sigmaPSD' : noise power spectral density (M x N double nonnegative array)
#   'profile' : 'np' --> Normal Profile
#               'lc' --> Fast Profile (slightly lower quality)
#   'blockMatches' : Tuple (HT, Wiener), with either value either:
#                      - False : Do not save blockmatches for phase
#                      - True : Save blockmatches for phase
#                      - Pre-computed block-matching array returned by a previous call with [True]
#                        Such as y_est, matches = BM3D(z, sigmaPSD, profile, (True, True))
#                        y_est2 = BM3D(z2, sigmaPSD, profile, matches);
#  OUTPUT:
#      'y_est'  denoised image  (M x N double array)
#      'y_est', ('blocks_ht', 'blocks_wie') denoised image, plus HT and Wiener blockmatches in a tuple,
#                                           if any storeBM values are set to True
#                                           (or [0] for missing block array, if one calculated)
#
#
#  BASIC SIMULATION EXAMPLES:
#
#     Case 1)
#
#      # Read a grayscale noise-free image
#
#      y=im2double(imread('cameraman.tif'))
#
#      # Generate noisy observations corrupted by additive colored random noise generated as convution of AWGN against with kernel 'k'
#
#      k=[-12-1]*[1 4 1]/100   # e.g., a diagonal kernel
#      z=y+imfilter(randn(size(y)),k(end:-1:1,end:-1:1),'circular')
#
#      # define 'sigmaPSD' from the kernel 'k'
#
#      sigmaPSD=abs(fft2(k,size(z,1),size(z,2))).^2*numel(z)
#
#      # Denoise 'z'
#      y_est = BM3D_COL2(z, sigmaPSD)
#
#
#     Case 2)
#
#      # Read a grayscale noise-free image
#
#      y=im2double(imread('cameraman.tif'))
#
#      # Generate noisy observations corrupted by additive colored random noise generated as convution of AWGN against with kernel 'k'
#      [x2, x1]=meshgrid(ceil(-size(y,2)/2):ceil(size(y,2)/2)-1,ceil(-size(y,1)/2):ceil(size(y,1)/2)-1)
#      sigmaPSD=ifftshift(exp(-((x1/size(y,1)).^2+(x2/size(y,2)).^2)*10))*numel(y)/100
#      z=y+real(ifft2(fft2(randn(size(y))).*sqrt(sigmaPSD)/sqrt(numel(y))))
#
#      # Denoise 'z'
#      y_est = BM3D_COL2(z, sigmaPSD)
#
#     Case 3) If 'sigmaPSD' is a singleton, this value is taken as sigma and it is assumed that the noise is white variance sigma^2.
#
#      # Read a grayscale noise-free image
#
#      y=im2double(imread('cameraman.tif'))
#
#      # Generate noisy observations corrupted by additive white Gaussian noise with variance sigma^2
#      sigma=0.1
#      z=y+sigma*randn(size(y))
#
#      y_est = BM3D_COL2(z, sigma)
#
#      # or, equivalently,
#      sigmaPSD = ones(size(z))*sigma^2*numel(z)
#      y_est = BM3D_COL2(z, sigmaPSD)
#
#
##########################################################################
#
# Copyright (c) 2006-2018 Tampere University of Technology.
# All rights reserved.
# This work (software, material, and documentation) shall only
# be used for nonprofit noncommercial purposes.
# Any unauthorized use of this work for commercial or for-profit purposes
# is prohibited.
#
# AUTHORS:
#     Y. Mäkinen, L. Azzari, K. Dabov, A. Foi
#     email: alessandro.foi@tut.fi
#
##########################################################################

import numpy as np
import pywt
from scipy.fftpack import *
from scipy.linalg import *
from scipy.ndimage.filters import convolve, correlate
from scipy import signal
from .bm3d_c import BM3DCaller

def BM3D(z, sigmaPSD, profile='np', blockMatches=(False, False)):
    ####  Quality/complexity trade-off profile selection
    ####
    ####  'np' --> Normal Profile (balanced quality)
    ####  'lc' --> Low Complexity Profile (fast, lower quality)
    ####


    ##########################################################################
    #### Following are the parameters for the Normal Profile.
    ####

    #### Select transforms ('dct', 'dest', 'hadamard', 'eye' or anything that is listed by 'help wfilters'):
    transform_2D_HT_name = 'bior1.5'  # 'dct'# ## transform used for the HT filt. of size N1 x N1
    # transform_2D_HT_name = 'eye'  # 'dct'# ## transform used for the HT filt. of size N1 x N1
    transform_2D_Wiener_name = 'dct'  ## transform used for the Wiener filt. of size N1_wiener x N1_wiener
    # transform_2D_Wiener_name = 'bior1.5'  # 'dct'# ## transform used for the HT filt. of size N1 x N1
    # transform_2D_Wiener_name = 'eye'  ## transform used for the Wiener filt. of size N1_wiener x N1_wiener
    transform_3rd_dim_name = 'haar'  # 'dct'#    ## transform used in the 3rd dim, the same for HT and Wiener filt.

    Nf = 32  # domain size for FFT computations
    Kin = 4   # how many layers of var3D to calculate
    useM = 0  # [for internal testing]

    denoise_residual = False  # Perform residual thresholding and re-denoising
    residual_thr = 3

    #### Hard-thresholding (HT) parameters:
    N1 = 8  ## N1 x N1 is the block size used for the hard-thresholding (HT) filtering
    Nstep = 3  ## sliding step to process every next reference block
    N2 = 16  ## maximum number of similar blocks (maximum size of the 3rd dimension of a 3D array)
    Ns = 39  ## length of the side of the search neighborhood for full-search block-matching (BM), must be odd
    tau_match = 3000  ## threshold for the block-distance (d-distance)
    lambda_thr3D = 2.7  ## threshold parameter for the hard-thresholding in 3D transform domain
    beta = 2.0  ## parameter of the 2D Kaiser window used in the reconstruction
    gamma = 4.0  # blockmatching confidence interval

    #### Wiener filtering parameters:
    N1_wiener = 8
    Nstep_wiener = 3
    N2_wiener = 32
    Ns_wiener = 39
    tau_match_wiener = 400
    beta_wiener = 2.0

    if profile == 'lc':
        Nstep = 6
        Ns = 25
        Nstep_wiener = 5
        N2_wiener = 16
        Ns_wiener = 25

    # Profile 'vn' was proposed in
    #  Y. Hou, C. Zhao, D. Yang, and Y. Cheng, 'Comment on "Image Denoising by Sparse 3D Transform-Domain
    #  Collaborative Filtering"', accepted for publication, IEEE Trans. on Image Processing, July, 2010.
    # as a better alternative to that initially proposed in [1] (which is currently in profile 'vn_old')
    if profile == 'vn':
        N2 = 32
        Nstep = 4

        N1_wiener = 11
        Nstep_wiener = 6

        lambda_thr3D = 2.8
        tau_match_wiener = 3500

        Ns_wiener = 39

    # The 'vn_old' profile corresponds to the original parameters for strong noise proposed in [1].
    if profile == 'vn_old':
        transform_2D_HT_name = 'dct'

        N1 = 12
        Nstep = 4

        N1_wiener = 11
        Nstep_wiener = 6

        lambda_thr3D = 2.8
        lambda_thr2D = 2.0  # no longer used
        tau_match_wiener = 3500
        tau_match = 5000

        Ns_wiener = 39

    decLevel = 0  ## dec. levels of the dyadic wavelet 2D transform for blocks (0 means full decomposition, higher values decrease the dec. number)

    if profile == 'high':  ## this profile is not documented in [1]

        decLevel = 1
        Nstep = 2
        Nstep_wiener = 2
        lambda_thr3D = 2.5
        beta = 2.5
        beta_wiener = 1.5

    ##########################################################################
    ##########################################################################
    #### Note: touch below this point only if you know what you are doing!
    ##########################################################################

    if Nf > 0:
        tau_match = 3000  # Adjust for variance subtraction

    ##########################################################################
    #### Create transform matrices, etc.
    ####
    Tfor, Tinv = getTransfMatrix(N1, transform_2D_HT_name,
                                 decLevel, False)  ## get (normalized) forward and inverse transform matrices
    TforW, TinvW = getTransfMatrix(N1_wiener, transform_2D_Wiener_name,
                                   0, False)  ## get (normalized) forward and inverse transform matrices

    if not useM and (transform_3rd_dim_name == 'haar' or transform_3rd_dim_name[-3:] == '1.1'):
        ### If Haar is used in the 3-rd dimension, then a fast internal transform is used, thus no need to generate transform
        ### matrices.
        hadper_trans_single_den = {}
        inverse_hadper_trans_single_den = {}
    else:
        ### Create transform matrices. The transforms are later applied by
        ### matrix-vector multiplication for the 1D case.
        hadper_trans_single_den = {}
        inverse_hadper_trans_single_den = {}

        rangemax = np.ceil(np.log2(np.max([N2, N2_wiener]))) + 1
        for hpow in range(0, int(rangemax)):
            h = 2 ** hpow
            Tfor3rd, Tinv3rd = getTransfMatrix(h, transform_3rd_dim_name, 0, True)
            hadper_trans_single_den[h] = (Tfor3rd)
            inverse_hadper_trans_single_den[h] = (Tinv3rd.T)

    ##########################################################################
    #### 2D Kaiser windows used in the aggregation of block-wise estimates
    ####
    if beta_wiener == 2 and beta == 2 and N1_wiener == 8 and N1 == 8:  # hardcode the window function so that the signal processing toolbox is not needed by default
        Wwin2D = [[0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924],
                  [0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989],
                  [0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846],
                  [0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325],
                  [0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325],
                  [0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846],
                  [0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989],
                  [0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924]]

        Wwin2D_wiener = Wwin2D
    else:
        Wwin2D = np.transpose([np.kaiser(N1, beta)]) @ [np.kaiser(N1, beta)]  # Kaiser window used in the aggregation of the HT part
        Wwin2D_wiener = np.transpose([np.kaiser(N1_wiener, beta_wiener)]) @ [np.kaiser(N1_wiener, beta_wiener)]  # Kaiser window used in the aggregation of the Wiener filt. part

    Wwin2D = np.array(Wwin2D)
    Wwin2D_wiener = np.array(Wwin2D_wiener)

    ##########################################################################

    z = np.array(z)

    if len(z.shape) == 2:
         z = np.array([z])

    sigmaPSD = np.array(sigmaPSD)

    channel_count = len(z)

    blockMatchesHT, blockMatchesWie = blockMatches  # Break apart


    # Convert blockmatch args to array even if they're single value
    if type(blockMatchesHT) == bool:
        blockMatchesHT = np.array([blockMatchesHT], dtype=np.intc)
    if type(blockMatchesWie) == bool:
        blockMatchesWie = np.array([blockMatchesWie], dtype=np.intc)


    # Ensure PSD resized to Nf is usable

    psd_k_sz = [1 + 2 * (np.floor(0.5 * sigmaPSD.shape[0] / Nf)), 1 + 2 * (np.floor(0.5 * sigmaPSD.shape[1] / Nf))]
    psd_k = gaussianKernel([int(psd_k_sz[0]), int(psd_k_sz[1])], 1 + 2 * (np.floor(0.5 * sigmaPSD.shape[1] / Nf)) / 20)
    psd_k = psd_k / np.sum(psd_k)

    if Nf > 0 and sigmaPSD.size > 1:
        PSD_blur = correlate(sigmaPSD, psd_k, mode='wrap')
    else:
        PSD_blur = sigmaPSD

    k = gaussianKernel([9, 9], 2)

    ##########################################################################
    #### Step 1. Produce the basic estimate by HT filtering
    ####

    bm3dCaller = BM3DCaller()

    y_hat, ht_blocks = bm3dCaller.bm3d_thr_colored_noise(z, lambda_thr3D, Nstep, N1, N2,
                                                         tau_match * N1 * N1 / (255 * 255), (Ns - 1) / 2,
                                                         Tfor, Tinv.T, hadper_trans_single_den, inverse_hadper_trans_single_den,
                                                         np.ones((N1, N1)),
                                                         Wwin2D, PSD_blur, Nf, Kin, gamma, channel_count, blockMatchesHT)

    # print('Hard-thresholding stage completed')

    if denoise_residual:
        resid = fft2(z - y_hat)[0]
        cc = correlate(np.array(abs(resid) > residual_thr * np.sqrt(sigmaPSD), dtype=np.float), k, mode='wrap')

        # Threshold mask
        msk = (cc > 0.01)

        # Residual + PSD
        remains = np.real(ifft2(resid * msk))
        remains_PSD = sigmaPSD * msk

        # If we don't convolve here, we might get an empty PSD in Nf-size...
        remains_PSD = correlate(remains_PSD, psd_k, mode='wrap')


        # Re-filter
        y_hat, ht_blocks = bm3dCaller.bm3d_thr_colored_noise(y_hat + remains, lambda_thr3D, Nstep, N1, N2,
                                                             tau_match * N1 * N1 / (255 * 255), (Ns - 1) / 2,
                                                             Tfor, Tinv.T, hadper_trans_single_den,
                                                             inverse_hadper_trans_single_den,
                                                             np.ones((N1, N1)),
                                                             Wwin2D, remains_PSD, Nf, Kin, gamma, channel_count,
                                                             blockMatchesHT)


    #return y_hat[0]

    ##########################################################################
    #### Step 2. Produce the final estimate by Wiener filtering (using the
    ####  hard-thresholding initial estimate)
    ###

    y_hat, wie_blocks = bm3dCaller.bm3d_wie_colored_noise(z, y_hat, Nstep_wiener, N1_wiener, N2_wiener,
                                                          tau_match_wiener * N1_wiener * N1_wiener / (255 * 255), (Ns_wiener - 1) / 2,
                                                          TforW, TinvW.T, hadper_trans_single_den, inverse_hadper_trans_single_den,
                                                          np.ones((N1, N1)), Wwin2D_wiener, PSD_blur, Nf, Kin, channel_count,
                                                          blockMatchesWie)
    if denoise_residual:
        resid = fft2(z - y_hat)[0]
        cc = correlate(np.array(abs(resid) > residual_thr * np.sqrt(sigmaPSD), dtype=np.float), k, mode='wrap')

        # Threshold mask
        msk = (cc > 0.01)

        # Residual + PSD
        remains = np.real(ifft2(resid * msk))
        remains_PSD = sigmaPSD * msk

        # If we don't convolve here, we might get an empty PSD in Nf-size...
        remains_PSD = correlate(remains_PSD, psd_k, mode='wrap')

        # Re-filter
        y_hat, wie_blocks = bm3dCaller.bm3d_wie_colored_noise(y_hat + remains, y_hat, Nstep_wiener, N1_wiener, N2_wiener,
                                                              tau_match_wiener * N1_wiener * N1_wiener / (255 * 255),
                                                              (Ns_wiener - 1) / 2,
                                                              TforW, TinvW.T, hadper_trans_single_den,
                                                              inverse_hadper_trans_single_den,
                                                              np.ones((N1, N1)), Wwin2D_wiener, remains_PSD, Nf, Kin,
                                                              channel_count,
                                                              blockMatchesWie)

    # print('Wiener-filtering stage completed')

    if channel_count == 1:  # Remove useless dimension if only single output
        y_hat = y_hat[0]

    if blockMatchesHT[0] == 1 and blockMatchesWie[0] != 1:  # We computed & want to return block-matches for HT
        return y_hat, (ht_blocks, np.zeros(1, dtype=np.intc))
    if blockMatchesHT[0] == 1:  # Both
        return y_hat, (ht_blocks, wie_blocks)
    if blockMatchesHT[0] != 1 and blockMatchesWie[0] == 1:  # Only wiener
        return y_hat, (np.zeros(1, dtype=np.intc), wie_blocks)

    return y_hat


##########################################################################
# Some auxiliary functions
##########################################################################

def gaussianKernel(size, std):
    g1d = signal.gaussian(size[0], std=std).reshape(size[0], 1)
    g1d2 = signal.gaussian(size[1], std=std).reshape(size[1], 1)

    g2d = np.outer(g1d, g1d2)
    return g2d

def getTransfMatrix(N, transform_type, dec_levels=0, flip_hardcoded=False):
    #
    # Create forward and inverse transform matrices, which allow for perfect
    # reconstruction. The forward transform matrix is normalized so that the
    # l2-norm of each basis element is 1.
    #
    # [Tforward, Tinverse] = getTransfMatrix (N, transform_type, dec_levels)
    #
    #  INPUTS:
    #
    #   N               --> Size of the transform (for wavelets, must be 2^K)
    #
    #   transform_type  --> 'dct', 'dest', 'hadamard', or anything that is
    #                       listed by 'help wfilters' (bi-orthogonal wavelets)
    #                       'DCrand' -- an orthonormal transform with a DC and all
    #                       the other basis elements of random nature
    #
    #   dec_levels      --> If a wavelet transform is generated, this is the
    #                       desired decomposition level. Must be in the
    #                       range [0, log2(N)-1], where "0" implies
    #                       full decomposition.
    #
    #  OUTPUTS:
    #
    #   Tforward        --> (N x N) Forward transform matrix
    #
    #   Tinverse        --> (N x N) Inverse transform matrix
    #

    if N == 1:
        Tforward = 1
    elif transform_type == 'hadamard':
        Tforward = hadamard(N)
    elif N == 8 and transform_type == 'bior1.5':  # hardcoded transform so that the wavelet toolbox is not needed to generate it
        Tforward = [[0.343550200747110, 0.343550200747110, 0.343550200747110, 0.343550200747110, 0.343550200747110,
                     0.343550200747110, 0.343550200747110, 0.343550200747110],
                    [-0.225454819240296, -0.461645582253923, -0.461645582253923, -0.225454819240296, 0.225454819240296,
                     0.461645582253923, 0.461645582253923, 0.225454819240296],
                    [0.569359398342840, 0.402347308162280, -0.402347308162280, -0.569359398342840, -0.083506045090280,
                     0.083506045090280, -0.083506045090280, 0.083506045090280],
                    [-0.083506045090280, 0.083506045090280, -0.083506045090280, 0.083506045090280, 0.569359398342840,
                     0.402347308162280, -0.402347308162280, -0.569359398342840],
                    [0.707106781186550, -0.707106781186550, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0.707106781186550, -0.707106781186550, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.707106781186550, -0.707106781186550, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0.707106781186550, -0.707106781186550]]
        if flip_hardcoded:
            Tforward = np.array(Tforward).T

    elif N == 8 and transform_type == 'dct':  # hardcoded transform so that the signal processing toolbox is not needed to generate it
        Tforward = [[0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274,
                     0.353553390593274, 0.353553390593274, 0.353553390593274],
                    [0.490392640201615, 0.415734806151273, 0.277785116509801, 0.097545161008064, -0.097545161008064,
                     -0.277785116509801, -0.415734806151273, -0.490392640201615],
                    [0.461939766255643, 0.191341716182545, -0.191341716182545, -0.461939766255643, -0.461939766255643,
                     -0.191341716182545, 0.191341716182545, 0.461939766255643],
                    [0.415734806151273, -0.097545161008064, -0.490392640201615, -0.277785116509801, 0.277785116509801,
                     0.490392640201615, 0.097545161008064, -0.415734806151273],
                    [0.353553390593274, -0.353553390593274, -0.353553390593274, 0.353553390593274, 0.353553390593274,
                     -0.353553390593274, -0.353553390593274, 0.353553390593274],
                    [0.277785116509801, -0.490392640201615, 0.097545161008064, 0.415734806151273, -0.415734806151273,
                     -0.097545161008064, 0.490392640201615, -0.277785116509801],
                    [0.191341716182545, -0.461939766255643, 0.461939766255643, -0.191341716182545, -0.191341716182545,
                     0.461939766255643, -0.461939766255643, 0.191341716182545],
                    [0.097545161008064, -0.277785116509801, 0.415734806151273, -0.490392640201615, 0.490392640201615,
                     -0.415734806151273, 0.277785116509801, -0.097545161008064]]
        if flip_hardcoded:
            Tforward = np.array(Tforward).T

    elif N == 11 and transform_type == 'dct':  # hardcoded transform so that the signal processing toolbox is not needed to generate it
        Tforward = [[0.301511344577764,  0.301511344577764,  0.301511344577764,  0.301511344577764,  0.301511344577764,  0.301511344577764,  0.301511344577764,  0.301511344577764,  0.301511344577764,  0.301511344577764,  0.301511344577764],
                   [0.422061280946316,  0.387868386059133,  0.322252701275551,  0.230530019145232,  0.120131165878581,  -8.91292406723889e-18,  -0.120131165878581,  -0.230530019145232,  -0.322252701275551,  -0.387868386059133,  -0.422061280946316],
                   [0.409129178625571,  0.279233555180591,  0.0606832509357945,  -0.177133556713755,  -0.358711711672592,  -0.426401432711221,  -0.358711711672592,  -0.177133556713755,  0.0606832509357945,  0.279233555180591,  0.409129178625571],
                   [0.387868386059133,  0.120131165878581,  -0.230530019145232,  -0.422061280946316,  -0.322252701275551,  1.71076608154014e-17,  0.322252701275551,  0.422061280946316,  0.230530019145232,  -0.120131165878581,  -0.387868386059133],
                   [0.358711711672592,  -0.0606832509357945,  -0.409129178625571,  -0.279233555180591,  0.177133556713755,  0.426401432711221,  0.177133556713755,  -0.279233555180591,  -0.409129178625571,  -0.0606832509357945,  0.358711711672592],
                   [0.322252701275551,  -0.230530019145232,  -0.387868386059133,  0.120131165878581,  0.422061280946316,  -8.13580150049806e-17,  -0.422061280946316,  -0.120131165878581,  0.387868386059133,  0.230530019145232,  -0.322252701275551],
                   [0.279233555180591,  -0.358711711672592,  -0.177133556713755,  0.409129178625571,  0.0606832509357945,  -0.426401432711221,  0.0606832509357944,  0.409129178625571,  -0.177133556713755,  -0.358711711672592,  0.279233555180591],
                   [0.230530019145232,  -0.422061280946316,  0.120131165878581,  0.322252701275551,  -0.387868386059133,  -2.87274927630557e-18,  0.387868386059133,  -0.322252701275551,  -0.120131165878581,  0.422061280946316,  -0.230530019145232],
                   [0.177133556713755,  -0.409129178625571,  0.358711711672592,  -0.0606832509357945,  -0.279233555180591,  0.426401432711221,  -0.279233555180591,  -0.0606832509357944,  0.358711711672592,  -0.409129178625571,  0.177133556713755],
                   [0.120131165878581,  -0.322252701275551,  0.422061280946316,  -0.387868386059133,  0.230530019145232,  2.03395037512452e-17,  -0.230530019145232,  0.387868386059133,  -0.422061280946316,  0.322252701275551,  -0.120131165878581],
                   [0.0606832509357945,  -0.177133556713755,  0.279233555180591,  -0.358711711672592,  0.409129178625571,  -0.426401432711221,  0.409129178625571,  -0.358711711672592,  0.279233555180591,  -0.177133556713755,  0.0606832509357945]]
        if flip_hardcoded:
            Tforward = np.array(Tforward).T

    elif N == 8 and transform_type == 'dest':  # hardcoded transform so that the PDE toolbox is not needed to generate it
        Tforward = [[0.161229841765317, 0.303012985114696, 0.408248290463863, 0.464242826880013, 0.464242826880013,
                     0.408248290463863, 0.303012985114696, 0.161229841765317],
                    [0.303012985114696, 0.464242826880013, 0.408248290463863, 0.161229841765317, -0.161229841765317,
                     -0.408248290463863, -0.464242826880013, -0.303012985114696],
                    [0.408248290463863, 0.408248290463863, 0, -0.408248290463863, -0.408248290463863, 0,
                     0.408248290463863, 0.408248290463863],
                    [0.464242826880013, 0.161229841765317, -0.408248290463863, -0.303012985114696, 0.303012985114696,
                     0.408248290463863, -0.161229841765317, -0.464242826880013],
                    [0.464242826880013, -0.161229841765317, -0.408248290463863, 0.303012985114696, 0.303012985114696,
                     -0.408248290463863, -0.161229841765317, 0.464242826880013],
                    [0.408248290463863, -0.408248290463863, 0, 0.408248290463863, -0.408248290463863, 0,
                     0.408248290463863, -0.408248290463863],
                    [0.303012985114696, -0.464242826880013, 0.408248290463863, -0.161229841765317, -0.161229841765317,
                     0.408248290463863, -0.464242826880013, 0.303012985114696],
                    [0.161229841765317, -0.303012985114696, 0.408248290463863, -0.464242826880013, 0.464242826880013,
                     -0.408248290463863, 0.303012985114696, -0.161229841765317]]
        if flip_hardcoded:
            Tforward = np.array(Tforward).T

    elif transform_type == 'dct':
        Tforward = dct(np.eye(N), norm='ortho')
    elif transform_type == 'eye':
        Tforward = np.eye(N)
    elif transform_type == 'dest':
        Tforward = dest(np.eye(N), norm='ortho')
    elif transform_type == 'DCrand':
        x = np.random.normal(N)
        x[:, 0] = np.ones(len(x[:, 0]))
        Q, _, _ = np.linalg.qr(x)
        if Q[0] < 0:
            Q = -Q

        Tforward = Q.T

    else:  ## a wavelet decomposition supported by 'wavedec'
        ### Set periodic boundary conditions, to preserve bi-orthogonality
        Tforward = np.zeros((N, N))
        for i in range(0, N):
            Tforward[:, i] = pywt.wavedec(np.roll([1, np.zeros(1, N - 1)], [dec_levels, i - 1]), level=np.log2(N),
                                          wavelet=transform_type, mode='periodical')  ## construct transform matrix

    Tforward = np.array(Tforward)
    ### Normalize the basis elements
    if not ((N == 8) and transform_type == 'bior1.5'):
        try:
            Tforward = (Tforward.T @ np.diag(np.sqrt(1. / sum(Tforward ** 2, 0)))).T
        except TypeError: # Tforward was not an array...
            pass


    ### Compute the inverse transform matrix
    try:
        Tinverse = np.linalg.inv(Tforward)
    except:
        Tinverse = np.array(1)

    return Tforward, Tinverse
