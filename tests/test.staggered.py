# this file is modified from test.dslash.qcu.py
# 
import os
import sys
from time import perf_counter
from enum import Enum
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

test_dir = os.path.dirname(os.path.abspath(__file__))

from pyquda import init, core, quda, mpi, pyqcu as qcu
from pyquda.enum_quda import QudaParity
from pyquda.field import LatticeFermion
from pyquda.utils import gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nd, Ns = 4, 1
Nc = 3
latt_size = [4, 4, 4, 4] # lattice description
grid_size = [1, 1, 1, 1]     # process description


Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size

latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]

Lx, Ly, Lz, Lt = latt_size

xi_0, nu = 1, 1
mass=-3.5
# mass=0
coeff_r, coeff_t = 0, 0 #wilson


mpi.init(grid_size)

param = qcu.QcuParam()
grid = qcu.QcuGrid()
param.lattice_size = latt_size
grid.grid_size = grid_size

half_prec = 0
float_prec = 1
double_prec = 2

precision_table = ['half', 'float', 'double']


class QcuStaggeredPhase (Enum) :
    kQcuStaggeredPhaseNo = 0
    kQcuStaggeredPhaseMilc = 1
    kQcuStaggeredPhaseCps = 2
    kQcuStaggeredPhaseTifr = 3
class QcuDslashType (Enum) :
    kQcuWilsonDslash = 0
    kQcuStaggeredDslash = 1
class QcuParity (Enum) :
    kQcuEven = 0
    kQcuOdd = 1
class QcuDagger (Enum) :
    kQcuNoDagger = 0
    kQcuDagger = 1
class QcuPrecision (Enum) :
    kQcuHalf = 0
    kQcuSingle = 1
    kQcuDouble = 2


def get_phase (dim, x, y, z, t, L_t, staggered_phase, t_boundary = 1.0) :
    phase = 1.0
    if staggered_phase == QcuStaggeredPhase.kQcuStaggeredPhaseMilc :
        if dim == 0:
            phase = (1.0 - 2.0 * (t % 2))
        elif dim == 1:
            phase = (1.0 - 2.0 * ((t + x) % 2))
        elif dim == 2:
            phase = (1.0 - 2.0 * ((t + x + y) % 2))
        elif dim == 3:
            phase = 1.0 if t != L_t - 1 else t_boundary # (t == Lt-1) ? t_boundary : 1.0;

    elif staggered_phase == QcuStaggeredPhase.kQcuStaggeredPhaseTifr:
        if dim == 0 :
            phase = (1.0 - 2.0 * ((3 + t + z + y) % 2))
        elif dim == 1 :
            phase = (1.0 - 2.0 * ((2 + t + z) % 2) )
        elif dim == 2 :
            phase = (1.0 - 2.0 * ((1 + t) % 2) )
        elif dim == 3 :
            phase = t_boundary if t == L_t - 1 else 1.0
    elif staggered_phase == QcuStaggeredPhase.kQcuStaggeredPhaseCps:
        if dim == 0:
            phase = 1.0
        elif dim == 1:
            phase = 1.0 - 2.0 * ((1 + x) % 2)
            # phase = (1.0 - 2.0 * (x 
        elif dim == 2:
            phase = (1.0 - 2.0 * ((1 + x + y) % 2))
        elif dim == 3:
            sgn = 1.0 if t != L_t - 1 else t_boundary
            phase = (1.0 - 2.0 * ((1 + x + y + z) % 2)) * sgn
    return phase

def staggered_dslash (f_out: cp.array, gauge: cp.ndarray, f_in: cp.ndarray, latt_size: list, staggered_phase: QcuStaggeredPhase) -> cp.ndarray :
    # precond_latt_size = [Lx // 2, Ly, Lz, Lt]
    print(f'f_out.shape: {f_out.shape}')
    print(f'gauge.shape: {gauge.shape}')
    print(f'f_in.shape: {f_in.shape}')
    print(f'latt_size: {latt_size}')
    print(f'staggered_phase: {staggered_phase}')
    for parity in range(2):
        for t in range(latt_size[3]):
            for z in range(latt_size[2]):
                for y in range(latt_size[1]):
                    for x in range(latt_size[0] // 2):
                        f_out[parity, t, z, y, x] = 0
                        for nu in range(4):
                            origin_x = 2 * x + ((y + z + t) % 2 != parity)
                            phase = get_phase(nu, origin_x, y, z, t, Lt, staggered_phase)
                            # backward
                            coord = [x, y, z, t]
                            if nu == 0:
                                coord[0] = ((origin_x - 1 + latt_size[0]) % latt_size[0]) // 2
                            else:
                                coord[nu] = (coord[nu] + latt_size[nu] - 1) % latt_size[nu]
                            c_x, c_y, c_z, c_t = coord
                            U = gauge[nu, 1-parity, c_t, c_z, c_y, c_x].conj().T
                            in_f = f_in[1-parity, c_t, c_z, c_y, c_x]
                            f_out[parity, t, z, y, x] += phase * (U @ in_f.T).T
                            if parity == 0 and x == 1 and y == 1 and z == 0 and t == 0:
                                print(f'dim: {nu}, phase: {phase}, staggered_phase: {staggered_phase}, {1.0 - 2.0 * (origin_x % 2)}')
                                print(f'bwd,in_f: {in_f}')
                                print(f'bwd,U: {U}')
                            # if parity == 0 and x == 0 and y == 0 and z == 0 and t == 0:
                            #     print(f'bwd f_out[parity, t, z, y, x]: {f_out[parity, t, z, y, x]}')
                            # forward
                            coord = [x, y, z, t]
                            if nu == 0:
                                coord[0] = ((origin_x + 1) % latt_size[0]) // 2
                            else:
                                coord[nu] = (coord[nu] + 1) % latt_size[nu]
                            c_x, c_y, c_z, c_t = coord
                            U = gauge[nu, parity, t, z, y, x]
                            in_f = f_in[1-parity, c_t, c_z, c_y, c_x]

                            if parity == 0 and x == 1 and y == 1 and z == 0 and t == 0:
                                print(f'fwd,in_f: {in_f}')
                                # print(f'nu = {nu}, parity: {parity}, t: {t}, z: {z}, y: {y}, x: {x}')
                                # print(f'{gauge[nu, parity, t, z, y, x]}')
                                print(f'fwd,U: {U}')
                            # print(f'phase: {phase}')
                            # print(f'in_f: {in_f.shape}')
                            # print(f'U: {U.shape}')
                            # tmp = phase * in_f * U
                            # print(f'tmp.shape: {tmp.shape}')
                            # print(f'f_out[parity, t, z, y, x].shape: {f_out[parity, t, z, y, x].shape}')
                            f_out[parity, t, z, y, x] += phase * (U @ in_f.T).T


                            if parity == 0 and x == 1 and y == 1 and z == 0 and t == 0:
                                print(f'fwd f_out[parity, t, z, y, x]: {f_out[parity, t, z, y, x]}')


def compare_result(quda_result, qcu_result):
    same = []
    diff = []
    print(f'quda_result.data.shape: {quda_result.data.shape}')
    print(f'qcu_result.data.shape: {qcu_result.data.shape}')
    for parity in range(2):
        for t in range(Lt):
            for z in range(Lz):
                for y in range(Ly):
                    for x in range(Lx // 2):
                        point_diff = cp.linalg.norm(quda_result.data[parity, t, z, y, x] - qcu_result.data[parity, t, z, y, x])
                        if point_diff < 1e-6:
                            same.append(point_diff)
                        else:
                            diff.append(point_diff)
    print(f'same: {len(same)}, diff: {len(diff)}, total: {len(same) + len(diff)}')


def test_mpi(round, my_m_input, warm_flag = False):
    staggered_phase = QcuStaggeredPhase.kQcuStaggeredPhaseTifr

    from pyquda.mpi import (
        comm, rank, size, grid, coord, gpuid
    )

    p_mrhs = [LatticeFermion(latt_size, Nc, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)) \
            for i in range(my_m_input)]
    # p_mrhs = [LatticeStagg(latt_size, Nc, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)) \
            # for i in range(my_m_input)]
    print(f'{p_mrhs[0].data.shape}')
    # p_mrhs = [LatticeFermion(latt_size, Nc, cp.ones((Lt, Lz, Ly, Lx, Ns, Nc * 2)).view(cp.complex128)) \
    #     for i in range(my_m_input)]

    quda_Mp_mrhs = [LatticeFermion(latt_size, Nc) for i in range(my_m_input)]
    qcu_Mp_mrhs = [LatticeFermion(latt_size, Nc) for i in range(my_m_input)]

    print(f'shape of quda_Mp_mrhs[0].data: {quda_Mp_mrhs[0].data.shape}')

    
    # dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False)
    # dslash.invert_param.dslash_type = core.enum_quda.QudaDslashType.QUDA_STAGGERED_DSLASH
    # dslash.gauge_param.staggered_phase_type = core.enum_quda.QudaStaggeredPhase.QUDA_STAGGERED_PHASE_CPS
    # dslash.gauge_param.reconstruct = core.enum_quda.QudaReconstructType.QUDA_RECONSTRUCT_NO
    # quda.staggered
    U = gauge_utils.gaussGauge(latt_size, 0)
    staggered_dslash(quda_Mp_mrhs[0].data, U.data, p_mrhs[0].data, latt_size, staggered_phase)
    # print(f'quda_Mp_mrhs[0].data = \n{quda_Mp_mrhs[0].data[0, 0, 0, 0, 0]}')
    
    # dslash.loadGauge(U)
    cp.cuda.runtime.deviceSynchronize()
    # print(f'Gauge[0] = \n{U.data[0, 0, 0, 0, 0, 0]}')
    # print(f'fermion input[0] = \n{p_mrhs[0].data[1, 0, 0, 0, 0]}')

    # t1 = perf_counter()
    # for i in range(my_m_input):
    #     quda.dslashQuda(quda_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    #     quda.dslashQuda(quda_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    # cp.cuda.runtime.deviceSynchronize()
    # t2 = perf_counter()
    # quda_dslash_time = t2 - t1

    #my code 
    qcu.getDslash(QcuDslashType.kQcuStaggeredDslash.value, mass, 0) # 0----WILSON
    qcu.loadQcuGauge(U.data_ptr, QcuPrecision.kQcuDouble.value)		# 2---double 1--float 0---half
    qcu.setStaggeredPhase(staggered_phase.value)
    cp.cuda.runtime.deviceSynchronize()

    t1 = perf_counter()
    for i in range(my_m_input):
        qcu.pushBackFermions(qcu_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr)
    qcu.start_dslash(QcuParity.kQcuEven.value, QcuDagger.kQcuNoDagger.value)	# param1 : parity  param2: dagger

    for i in range(my_m_input):
        qcu.pushBackFermions(qcu_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr)
    qcu.start_dslash(QcuParity.kQcuOdd.value, QcuDagger.kQcuNoDagger.value)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_dslash_time = t2 - t1

    # if (not warm_flag):
    #     print(f"Quda dslash: {quda_dslash_time}sec \nQcu dslash:  {qcu_dslash_time} sec")
    
    if mpi.rank == 0:
        print(f'quda_res = \n{quda_Mp_mrhs[0].data[0, 0, 0, 1, 1]}')
        print(f'qcu_res = \n{qcu_Mp_mrhs[0].data[0, 0, 0, 1, 1]}')
        # print(f'gauge = \n{U.data[0, 1, 0, 0, 1, 0]}')
    '''
    average_difference = cp.sum(cp.array([cp.linalg.norm(quda_Mp_mrhs[i].data - qcu_Mp_mrhs[i].data) / cp.linalg.norm(quda_Mp_mrhs[i].data) \
                for i in range(my_m_input)])) / my_m_input
    print(f'rank {rank}, average difference: , {average_difference}')
    compare_result(quda_Mp_mrhs[0], qcu_Mp_mrhs[0])
    '''
    return 0, 0 # quda_dslash_time, qcu_dslash_time


def test_dslash(my_n_color, my_m_input, input_prec, dslash_prec, quda_average_time, qcu_average_time, warmup_flag = False)->int:
    qcu.initGridSize(grid, param, my_n_color, my_m_input, input_prec, dslash_prec)

    total_quda_time = 0
    total_qcu_time = 0

    if (not warmup_flag):
        print(f'=========== mrhs = {my_m_input} condition begin ===========')
    iteration = 1
    for i in range(iteration) :
        quda_time, qcu_time = test_mpi(i, my_m_input)
        total_quda_time += quda_time
        total_qcu_time += qcu_time

    if (not warmup_flag):
        quda_average_time.append(total_quda_time / iteration)
        qcu_average_time.append(total_qcu_time / iteration)
        print(f'=========== mrhs = {my_m_input} condition end ===========')

    qcu.finalizeQcu()
    cp.cuda.runtime.deviceSynchronize()

if __name__ == '__main__' :
    # _ = input()
    max_input = 1
    # my_n_color = Nc

    # operations_per_point = (2 * Nd * Nc * Ns) + (2 * Nd * Ns / 2 * (8 * Nc-2)*Nc) + ((2 * Nd - 1) * 2 * Nc * Ns)
    # operations_per_dslash = operations_per_point * Lx * Ly * Lz * Lt

    my_input_prec  = double_prec
    my_dslash_prec = double_prec

    quda_average_time = []
    qcu_average_time  = []

    # warm up
    # test_dslash(Nc, 1, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
    #     quda_average_time = quda_average_time, qcu_average_time = qcu_average_time, warmup_flag=True)
    # warm up end

    for my_m_input in range(1, max_input+1):
        test_dslash(Nc, my_m_input, input_prec=my_input_prec, dslash_prec=my_dslash_prec, quda_average_time = quda_average_time, qcu_average_time = qcu_average_time)