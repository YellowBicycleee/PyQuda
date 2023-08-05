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
                    # data_new[:, t, z, y, 0::2] = data_cb2[:, 0, t, z, y, :]
                    # data_new[:, t, z, y, 1::2] = data_cb2[:, 1, t, z, y, :]
                else:
                    data_new[:, 1, t, z, y, :] = data_cb2[:, t, z, y, 0::2]
                    data_new[:, 0, t, z, y, :] = data_cb2[:, t, z, y, 1::2]
                    # data_new[:, t, z, y, 1::2] = data_cb2[:, 0, t, z, y, :]
                    # data_new[:, t, z, y, 0::2] = data_cb2[:, 1, t, z, y, :]
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

    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)

    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, U_seed)
    dslash.loadGauge(U)

    # Load a from p and allocate b
    a = LatticeFermion(latt_size, cp.asarray(core.cb2(p, [0, 1, 2, 3])))
    b = LatticeFermion(latt_size)
    cp.cuda.runtime.deviceSynchronize()


   # Dslash a = b
#    quda.dslashQuda(b.even_ptr, a.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
#    quda.dslashQuda(b.odd_ptr, a.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    t2 = timeit.timeit(lambda: quda.dslashQuda(b.even_ptr, a.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY), number=1)
    t3 = timeit.timeit(lambda: quda.dslashQuda(b.odd_ptr, a.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY), number=1)
    # Save b to Mp
    Mp[:] = b.lexico()

    # Return gauge as a ndarray with shape (Nd, Lt, Lz, Ly, Lx, Ns, Ns)
    return U.lexico(), t2+t3

Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
# p = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
# p[0, 0, 0, 0, 0, 1] = 1
shape = (Lt, Lz, Ly, Lx, Ns, Nc)
p = np.random.randn(*shape, 2).view(np.complex128).reshape(shape)
U,_ = applyDslash(Mp, p, 0)

Mp1 = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
param = pyqcu.QcuParam()
param.lattice_size = latt_size

p_pre = vector_eo_pre(p)
Mp_pre = vector_eo_pre(Mp1)
U_pre = gauge_eo_pre(U)
pyqcu.dslashQcuEO(Mp_pre[0], p_pre[1], U, param, 0)
pyqcu.dslashQcuEO(Mp_pre[1], p_pre[0], U, param, 1)
my_res = vector_lexico(Mp_pre)
print(my_res[0,0,0,1])
print(Mp[0,0,0,1])
# U_lexico = gauge_lexico(U_pre)
# print('diff = ', np.linalg.norm(U_lexico - U))
'----------------------------------------'

print(np.linalg.norm(Mp[0,0,0,1] - my_res[0,0,0,1]))
print(np.linalg.norm(Mp[0,0,1,0] - my_res[0,0,1,0]))
print(np.linalg.norm(Mp[0,1,0,0] - my_res[0,1,0,0]))
print(np.linalg.norm(Mp[1,0,0,0] - my_res[1,0,0,0]))
print(np.linalg.norm(Mp - my_res) / np.linalg.norm(Mp))
for t in range(0, Lt):
    for z in range(0, Lz):
        for y in range(0, Ly):
            for x in range(0, Lx):
                if np.linalg.norm(Mp[t,z,y,x] - my_res[t,z,y,x]) > 0.5:
                    print('(', x, ', ',y, ', ', z, ', ', t, ')')
#print(np.linalg.norm(Mp - Mp1)/np.linalg.norm(Mp))

# def compare(round):
#     print('===============round ', round, '======================')
#     shape = (Lt, Lz, Ly, Lx, Ns, Nc)
#     p = np.random.randn(*shape, 2).view(np.complex128).reshape(shape)
#     print(p.shape)
#     Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
#     U, time_quda = applyDslash(Mp, p, 0)
#     print('Quda dslash: = ', time_quda, 's')
#     if round == 0:
#         print('QUDA result of [0, 0, 0, 1]')
#         print(Mp[0, 0, 0, 1])
#     Mp1 = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
#     param = pyqcu.QcuParam()
#     param.lattice_size = latt_size
#     t = timeit.timeit(lambda: pyqcu.dslashQcu(Mp1, p, U, param), number=1)
#     print('my_dslash: = ', t, 's')
#     if round == 0:
#         print('My result of [0, 0, 0, 1]')
#         print(Mp1[0, 0, 0, 1])
#     print('difference: ', np.linalg.norm(Mp-Mp1)/np.linalg.norm(Mp))

# # from time import perf_counter



# for i in range(0, 5):
#     compare(i)
