import os
import sys
import timeit

# test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))

import numpy as np

from pyquda import init, pyqcu

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
init()

Lx, Ly, Lz, Lt = 16, 16, 16, 32
Nd, Ns, Nc = 4, 4, 3
latt_size = [Lx, Ly, Lz, Lt]


def applyDslash(Mp, p, U_seed):
    import cupy as cp
    from pyquda import core, quda
    from pyquda.enum_quda import QudaParity
    from pyquda.field import LatticeFermion
    from pyquda.utils import gauge_utils

    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)

    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, U_seed)
    #dslash.loadGauge(U)
    t1 = timeit.timeit(lambda: dslash.loadGauge(U), number=1)

    # Load a from p and allocate b
    a = LatticeFermion(latt_size, cp.asarray(core.cb2(p, [0, 1, 2, 3])))
    b = LatticeFermion(latt_size)
    cp.cuda.runtime.devipesynchronize()


   # Dslash a = b
#    quda.dslashQuda(b.even_ptr, a.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
#    quda.dslashQuda(b.odd_ptr, a.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    t2 = timeit.timeit(lambda: quda.dslashQuda(b.even_ptr, a.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY), number=1)
    t3 = timeit.timeit(lambda: quda.dslashQuda(b.odd_ptr, a.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY), number=1)
    # Save b to Mp
    Mp[:] = b.lexico()

    # Return gauge as a ndarray with shape (Nd, Lt, Lz, Ly, Lx, Ns, Ns)
    return U.lexico(), t2+t3


p = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
p[0, 0, 0, 0, 0, 0] = 1
p[0, 0, 0, 0, 0, 1] = 1
p[0, 1, 0, 0, 0, 0] = 1
Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)

U,_ = applyDslash(Mp, p, 0)
print(Mp[0, 15, 0, 0])

Mp1 = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
param = pyqcu.QcuParam()
param.lattice_size = latt_size
pyqcu.dslashQcu(Mp1, p, U, param)
print(Mp1[0, 15, 0, 0])


print(np.linalg.norm(Mp - Mp1))
#print(np.linalg.norm(Mp - Mp1)/np.linalg.norm(Mp))

def compare(round):
    print('===============round ', round, '======================')
    shape = (Lt, Lz, Ly, Lx, Ns, Nc)
    p = np.random.randn(*shape, 2).view(np.complex128).reshape(shape)
    print(p.shape)
    Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
    U, time_quda = applyDslash(Mp, p, 0)
    print('Quda dslash: = ', time_quda, 's')
    if round == 0:
        print('QUDA result of [0, 0, 0, 1]')
        print(Mp[0, 0, 0, 1])
    Mp1 = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
    param = pyqcu.QcuParam()
    param.lattice_size = latt_size
    t = timeit.timeit(lambda: pyqcu.dslashQcu(Mp1, p, U, param), number=1)
    print('my_dslash: = ', t, 's')
    if round == 0:
        print('My result of [0, 0, 0, 1]')
        print(Mp1[0, 0, 0, 1])
    print('difference: ', np.linalg.norm(Mp-Mp1)/np.linalg.norm(Mp))

# from time import perf_counter



for i in range(0, 5):
    compare(i)
