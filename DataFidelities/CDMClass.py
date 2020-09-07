import numpy as np

from DataFidelities.DataClass import DataClass


class CDMClass(DataClass):

    def __init__(self, y, sigSize, A):
        self.y = y     # here y is a three three-dimensional matrix, and the first dimension index the sub-measurements
        self.A = A     # here A is a three-dimensional matrix, and the first dimension index the sub-measurements
        self.Nt = A.shape[0]
        self.sigSize = sigSize

    def eval(self, x):
        z = np.zeros(self.Nt, self.sigSize[0], self.sigSize[1])
        for meas_idx in range(self.Nt):
            zStoc = self.fmult(x, self.A[meas_idx,:,:])
            z[meas_idx,...] = zStoc
        d = np.linalg.norm(z.real.flatten('F') - self.y.flatten('F')) ** 2   # take out the real parts
        return d

    def gradStoc(self, x, meas_list):
        if not isinstance(meas_list, (list, np.ndarray)):
            print('meas_list for gradStoc should be list')
            exit()
        else:
            g = np.zeros(self.sigSize[0], self.sigSize[1])
            for meas_idx in meas_list:
                z = self.fmult(x, self.A[meas_idx,:,:])
                res = z - self.y[meas_idx,:,:] * z / abs(z)
                g  = g + self.ftran(res, self.A[meas_idx,:,:])
        return g.real       # only keep the real part

    def grad(self, x):
        g = self.gradStoc(x, list(range(self.Nt)))
        return g.real       # only keep the real part

    def draw(self, x):
        # plt.imshow(np.real(x),cmap='gray')
        pass
    
    @staticmethod
    def genMeas(sigSize, Nt):
        Areal = np.random.rand(Nt, sigSize[0], sigSize[1])
        Aimag = np.sqrt(1 - Areal ** 2)
        Areal = Areal * (-1) ** np.random.randint(10, size=Areal.shape)
        Aimag = Aimag * (-1) ** np.random.randint(10, size=Aimag.shape)
        A = Areal + 1j*Aimag
        return A

    @staticmethod
    def fmult(x, A):
        z =  np.fft.fft2(A * x) / np.sqrt(x.size)
        return z
    
    @staticmethod
    def ftran(z, A):
        x =  A.conj() * np.fft.ifft2(z) * np.sqrt(z.size)
        return x















