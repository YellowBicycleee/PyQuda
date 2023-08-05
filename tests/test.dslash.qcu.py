import os
import sys
import timeit
import numpy as np

from typing import List
from pyquda.field import lexico
def eo_pre(data: np.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    # Lx //= 2
    Npre = int(np.prod(shape[:axes[0]]))
    Nsuf = int(np.prod(shape[axes[-1] + 1:]))
    dtype = data.dtype if dtype is None else dtype
    # data_cb2 = data.reshape(Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf)
    data_cb2 = data.reshape(Npre, Lt, Lz, Ly, Lx, Nsuf)
    data_new = np.zeros((Npre, 2, Lt, Lz, Ly, Lx//2, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_new[:, 0, t, z, y, :] = data_cb2[:, t, z, y, 0::2]
                    data_new[:, 1, t, z, y, :] = data_cb2[:, t, z, y, 1::2]
                else:
                    data_new[:, 1, t, z, y, :] = data_cb2[:, t, z, y, 0::2]
                    data_new[:, 0, t, z, y, :] = data_cb2[:, t, z, y, 1::2]
    return data_new.reshape(*shape[:axes[0]], 2, Lt, Lz, Ly, Lx//2, *shape[axes[-1] + 1:])

def vector_lexico(vec): #(2, Lt, Lz, Ly, Lx, Nd, Nc)
    result = lexico(vec, [0, 1, 2, 3, 4])
    return result
def gauge_lexico(vec): #(Nd, 2, Lt, Lz, Ly, Lx, Nc, Nc)
    result = lexico(vec, [1, 2, 3, 4, 5])
    return result
def vector_eo_pre(vec):
    result = eo_pre(vec, [0, 1, 2, 3])
    return result
def gauge_eo_pre(vec):
    result = eo_pre(vec, [1, 2, 3, 4])
    return result




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
    from time import perf_counter
    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)

    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, U_seed)
    dslash.loadGauge(U)

    # Load a from p and allocate b
    a = LatticeFermion(latt_size, cp.asarray(core.cb2(p, [0, 1, 2, 3])))
    b = LatticeFermion(latt_size)
    cp.cuda.runtime.deviceSynchronize()
    t1 = perf_counter()
    # Dslash a = b
    quda.dslashQuda(b.even_ptr, a.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(b.odd_ptr, a.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    t2 = perf_counter()

    # Save b to Mp
    Mp[:] = b.lexico()

    # Return gauge as a ndarray with shape (Nd, Lt, Lz, Ly, Lx, Ns, Ns)
    return U.lexico(), t2-t1

# Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
# p = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
# p[0, 0, 0, 0, 0, 1] = 1
# shape = (Lt, Lz, Ly, Lx, Ns, Nc)
# p = np.random.randn(*shape, 2).view(np.complex128).reshape(shape)
# U,_ = applyDslash(Mp, p, 0)

# Mp1 = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
# param = pyqcu.QcuParam()
# param.lattice_size = latt_size

# p_pre = vector_eo_pre(p)
# Mp_pre = vector_eo_pre(Mp1)
# U_pre = gauge_eo_pre(U)
# pyqcu.dslashQcuEO(Mp_pre[0], p_pre[1], U_pre, param, 0)
# pyqcu.dslashQcuEO(Mp_pre[1], p_pre[0], U_pre, param, 1)
# my_res = vector_lexico(Mp_pre)
# print(my_res[0,0,0,1])
# print(Mp[0,0,0,1])
# res_p = vector_lexico(p_pre)
# print('p-resp: ', np.linalg.norm(p - res_p))
# U_lexico = gauge_lexico(U_pre)
# print('diff = ', np.linalg.norm(U_lexico - U))
'----------------------------------------'


#print(np.linalg.norm(Mp - Mp1)/np.linalg.norm(Mp))
shape = (Lt, Lz, Ly, Lx, Ns, Nc)

def compare(round):
    from time import perf_counter
    print('===============round ', round, '======================')
    # generate a vector p randomly
    p = np.random.randn(*shape, 2).view(np.complex128).reshape(shape)
    Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)      # store the result of QUDA
    # execute QUDA and get time
    U, time_quda = applyDslash(Mp, p, 0)
    print('Quda dslash: = ', time_quda, 's')

    # then execute my code
    Mp1 = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
    param = pyqcu.QcuParam()
    param.lattice_size = latt_size
    p_pre = vector_eo_pre(p)
    Mp_pre = vector_eo_pre(Mp1)
    U_pre = gauge_eo_pre(U)
    t1 = perf_counter()
    pyqcu.dslashQcuEO(Mp_pre[0], p_pre[1], U_pre, param, 0)
    pyqcu.dslashQcuEO(Mp_pre[1], p_pre[0], U_pre, param, 1)
    t2 = perf_counter()
    my_res = vector_lexico(Mp_pre)
    print('my_dslash total time: = ', t2-t1, 's')
    print('difference: ', np.linalg.norm(Mp-my_res)/np.linalg.norm(Mp))


for i in range(0, 5):
    compare(i)
