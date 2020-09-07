import ctypes
import os
import numpy as np
from numpy.ctypeslib import ndpointer
from . import print_mat

"""
This file calls the shared object c-functions with the given parameters.
Called from BM3D.py.
"""

class BM3DCaller:
    def __init__(self):
        self.__dll = ctypes.CDLL("./Regularizers/pybm3d/bm3d_thr.so")
        self.__threshold_noise = self.__dll.bm3d_threshold_colored_interface
        self.__threshold_noise.argtypes = [ctypes.POINTER(ctypes.c_double),  # AVG
                                           ctypes.c_float,  # thr3D
                                           ctypes.c_int,  # P
                                           ctypes.c_int,  # nm1
                                           ctypes.c_int,  # nm2
                                           ctypes.c_int,  # sz1
                                           ctypes.c_int,  # sz2
                                           ctypes.c_float,  # thrClose
                                           ctypes.c_int,  # searchWinSize
                                           ctypes.POINTER(ctypes.c_float),  # fMatN1
                                           ctypes.POINTER(ctypes.c_float),  # iMatN1
                                           ctypes.POINTER(ctypes.c_float),  # arbMat
                                           ctypes.POINTER(ctypes.c_float),  # arbMatInv
                                           ctypes.POINTER(ctypes.c_float),  # sigmas
                                           ctypes.POINTER(ctypes.c_double),  # WIN
                                           ctypes.POINTER(ctypes.c_float),  # PSD
                                           ctypes.c_int,  # Nf
                                           ctypes.c_int,  # Kin
                                           ctypes.c_float, # gamma
                                           ctypes.c_int,   # Channel count
                                           ctypes.POINTER(ctypes.c_int)  # Blockmatch info
                                           ]

        self.__dll2 = ctypes.CDLL("./Regularizers/pybm3d/bm3d_wie.so")
        self.__wiener_noise = self.__dll2.bm3d_wiener_colored_interface
        self.__wiener_noise.argtypes = [ctypes.POINTER(ctypes.c_double),  # Bn
                                           ctypes.POINTER(ctypes.c_float),  # AVG
                                           ctypes.c_int,  # P
                                           ctypes.c_int,  # nm1
                                           ctypes.c_int,  # nm2
                                           ctypes.c_int,  # sz1
                                           ctypes.c_int,  # sz2
                                           ctypes.c_float,  # thrClose
                                           ctypes.c_int,  # searchWinSize
                                           ctypes.POINTER(ctypes.c_float),  # fMatN1
                                           ctypes.POINTER(ctypes.c_float),  # iMatN1
                                           ctypes.POINTER(ctypes.c_float),  # arbMat
                                           ctypes.POINTER(ctypes.c_float),  # arbMatInv
                                           ctypes.POINTER(ctypes.c_float),  # sigmas
                                           ctypes.POINTER(ctypes.c_double),  # WIN
                                           ctypes.POINTER(ctypes.c_float),  # PSD
                                           ctypes.c_int,  # Nf
                                           ctypes.c_int,  # Kin
                                           ctypes.c_int,  # Channel count
                                           ctypes.POINTER(ctypes.c_int)  # Blockmatch info
                                           ]


        #self.__threshold_noise.errcheck = self.errcheck

    def conv_to_array(self, pyarr, type=ctypes.c_float):
        return (type * len(pyarr))(*pyarr)

    def flatten_transf(self, transf_dict, type=ctypes.c_float):
        total_list = []
        for key in sorted(transf_dict):
            flattened = list(transf_dict[key].flatten())
            total_list += flattened
        total_list = np.array(total_list)
        return (type * len(total_list))(*total_list)


    def bm3d_wie_colored_noise(self, Bn, AVG, P, nm1, nm2, thrClose, searchWinSize,
                               fMatN1, iMatN1, arb3dtransf, arbInv, sigmas, WIN, PSD, Nf, Kin, channel_count=1,
                               blockMatches=np.zeros(1, dtype=np.intc)):

        # Aaaand we need to convert everything into C-passable types
        # Flatten all arrays. Flipped since C was made for matlab :(
        AVG_shape = AVG.T.shape
        AVG = AVG.transpose(0, 2, 1).flatten()

        fMatN1 = fMatN1.T.flatten()
        iMatN1 = iMatN1.T.flatten()
        sigmas = sigmas.T.flatten()
        WIN = WIN.T.flatten()
        Bn = Bn.transpose(0, 2, 1).flatten()

        zero = 0
        pass_AVG = self.conv_to_array(AVG)
        pass_arb3dtransf = None if len(arb3dtransf) == 0 else self.flatten_transf(arb3dtransf)
        pass_P = ctypes.c_int(P)
        pass_nm1 = ctypes.c_int(nm1)
        pass_nm2 = ctypes.c_int(nm2)

        pass_sz1 = ctypes.c_int(AVG_shape[1])
        pass_sz2 = ctypes.c_int(AVG_shape[0])
        pass_thrClose = ctypes.c_float(thrClose)
        pass_searchWinSize = ctypes.c_int(int(searchWinSize))
        pass_fMatN1 = self.conv_to_array(fMatN1)
        pass_iMatN1 = self.conv_to_array(iMatN1)
        pass_arbMatInv = None if len(arb3dtransf) == 0 else self.flatten_transf(arbInv)
        pass_sigmas = self.conv_to_array(sigmas)
        pass_WIN = self.conv_to_array(WIN, ctypes.c_double)
        pass_PSD = self.conv_to_array(np.concatenate([[PSD.T.shape[2]], PSD.transpose(0, 2, 1).flatten()] if len(PSD.shape) > 2 else [[1], PSD.T.flatten()]) if type(PSD) == np.ndarray  and len(PSD) > 1 else [0, PSD])
        pass_Nf = ctypes.c_int(Nf)
        pass_Kin = ctypes.c_int(Kin)
        pass_Bn = self.conv_to_array(Bn, ctypes.c_double)
        pass_channel_count = ctypes.c_int(channel_count)
        pass_blockMatches = self.conv_to_array(blockMatches, ctypes.c_int)
        self.__wiener_noise.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))

        r = self.__wiener_noise(pass_Bn, pass_AVG, pass_P, pass_nm1, pass_nm2, pass_sz1, pass_sz2, pass_thrClose, pass_searchWinSize, pass_fMatN1,
                               pass_iMatN1, pass_arb3dtransf, pass_arbMatInv, pass_sigmas, pass_WIN, pass_PSD, pass_Nf, pass_Kin, pass_channel_count,
                               pass_blockMatches)

        # Get the contents of the returned array possibly with some type of magic
        # (get the address, add the offset, and for some reason requires a cast to a float pointer after that)
        r0 = r[0]
        holder = ctypes.POINTER(ctypes.c_float)
        returns = []

        width = AVG_shape[1]
        height = AVG_shape[0]
        for k in range(channel_count):
            retArr = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    retArr[i][j] = ctypes.cast(ctypes.addressof(r0)
                                               + ctypes.sizeof(ctypes.c_float)
                                               * (width * height * k + i * width + j), holder).contents.value
            returns.append(retArr.T)

        # Blockmatching stuff if needed
        bm_array = []
        if blockMatches[0] == 1:
            # Acquire the blockmatch data array...
            r0 = r[0]
            holder = ctypes.POINTER(ctypes.c_int)
            # The size of image returns
            startingPoint = ctypes.sizeof(ctypes.c_float) * (width * height * channel_count)

            # The first element of the BM contains its total size of int
            bm_array = np.zeros(ctypes.cast(ctypes.addressof(r0) + startingPoint, holder).contents.value, dtype=np.intc)
            for i in range(bm_array.size):
                bm_array[i] = ctypes.cast(ctypes.addressof(r0) + startingPoint + ctypes.sizeof(ctypes.c_int) * i, holder).contents.value

        return np.array(returns), bm_array



    def bm3d_thr_colored_noise(self, AVG, thr3D, P, nm1, nm2, thrClose, searchWinSize, fMatN1, iMatN1, arb3dtransf,
                               arbInv, sigmas, WIN, PSD, Nf, Kin, gamma, channel_count=1, blockMatches=np.zeros(1, dtype=np.intc)):
        # we need to convert everything into C-passable types
        # Flatten all arrays.
        AVG_shape = AVG.T.shape
        AVG = AVG.transpose(0, 2, 1).flatten()

        fMatN1 = fMatN1.T.flatten()
        iMatN1 = iMatN1.T.flatten()
        sigmas = sigmas.T.flatten()
        WIN = WIN.T.flatten()

        pass_AVG = self.conv_to_array(AVG, ctypes.c_double)
        pass_arb3dtransf = None if len(arb3dtransf) == 0 else self.flatten_transf(arb3dtransf)
        pass_P = ctypes.c_int(P)
        pass_nm1 = ctypes.c_int(nm1)
        pass_nm2 = ctypes.c_int(nm2)

        pass_sz1 = ctypes.c_int(AVG_shape[1])
        pass_sz2 = ctypes.c_int(AVG_shape[0])

        pass_thr3D = ctypes.c_float(thr3D)
        pass_thrClose = ctypes.c_float(thrClose)
        pass_searchWinSize = ctypes.c_int(int(searchWinSize))
        pass_fMatN1 = self.conv_to_array(fMatN1)
        pass_iMatN1 = self.conv_to_array(iMatN1)
        pass_arbMatInv = None if len(arb3dtransf) == 0 else self.flatten_transf(arbInv)
        pass_sigmas = self.conv_to_array(sigmas)
        pass_WIN = self.conv_to_array(WIN, ctypes.c_double)
        pass_PSD = self.conv_to_array(np.concatenate([[PSD.T.shape[2]], PSD.transpose(0, 2, 1).flatten()] if len(PSD.shape) > 2 else [[1], PSD.T.flatten()]) if type(PSD) == np.ndarray  and len(PSD) > 1 else [0, PSD])
        pass_Nf = ctypes.c_int(Nf)
        pass_Kin = ctypes.c_int(Kin)
        pass_gamma = ctypes.c_float(gamma)

        pass_channel_count = ctypes.c_int(channel_count)
        pass_blockMatches = self.conv_to_array(blockMatches, ctypes.c_int)

        self.__threshold_noise.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))

        r = self.__threshold_noise(pass_AVG, pass_thr3D, pass_P, pass_nm1, pass_nm2, pass_sz1, pass_sz2,
                               pass_thrClose, pass_searchWinSize, pass_fMatN1,
                               pass_iMatN1, pass_arb3dtransf, pass_arbMatInv, pass_sigmas,
                                   pass_WIN, pass_PSD, pass_Nf, pass_Kin, pass_gamma, pass_channel_count,
                                   pass_blockMatches)
        r0 = r[0]
        holder = ctypes.POINTER(ctypes.c_float)
        returns = []

        width = AVG_shape[1]
        height = AVG_shape[0]
        for k in range(channel_count):
            retArr = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    retArr[i][j] = ctypes.cast(ctypes.addressof(r0)
                                               + ctypes.sizeof(ctypes.c_float)
                                               * (width * height * k + i * width + j), holder).contents.value
            returns.append(retArr.T)

        # Blockmatching stuff if needed
        bm_array = []
        if blockMatches[0] == 1:
            # Acquire the blockmatch data array...
            r0 = r[0]
            holder = ctypes.POINTER(ctypes.c_int)
            # The size of image returns
            startingPoint = ctypes.sizeof(ctypes.c_float) * (width * height * channel_count)

            # The first element of the BM contains its total size of int
            bm_array = np.zeros(ctypes.cast(ctypes.addressof(r0) + startingPoint, holder).contents.value, dtype=np.intc)
            for i in range(bm_array.size):
                bm_array[i] = ctypes.cast(ctypes.addressof(r0) + startingPoint + ctypes.sizeof(ctypes.c_int) * i, holder).contents.value

        return np.array(returns), bm_array


