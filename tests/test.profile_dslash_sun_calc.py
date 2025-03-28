# this file is modified from test.dslash.qcu.py
# 
import os
import sys
from time import perf_counter

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

test_dir = os.path.dirname(os.path.abspath(__file__))

from pyquda import init, core, quda, mpi, pyqcu as qcu
from pyquda.enum_quda import QudaParity
from pyquda.field import LatticeFermion
from pyquda.utils import gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nd, Ns = 4, 4
Nc = 32
# latt_size = [4, 4, 4, 4] # lattice description
latt_size = [16, 8, 16, 16] # lattice description
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

class Precision :    
    kPrecisionHalf = 0
    kPrecisionSingle = 1
    kPrecisionDouble = 2

    @staticmethod
    def get_precision_name(precision):
        if precision == Precision.kPrecisionHalf:
            return 'half'
        elif precision == Precision.kPrecisionSingle:
            return 'single'
        elif precision == Precision.kPrecisionDouble:
            return 'double'
        else:
            raise ValueError(f"Invalid precision: {precision}")


'''
    计算Wilson费米子场操作的flop
'''
def get_wilson_flop_4dim (latt_desc: list, color: int) :
    operations_per_point = (2 * Nd * Nc * Ns) + (2 * Nd * Ns / 2 * (8 * Nc)*Nc) + ((2 * Nd) * 2 * Nc * Ns)
    vol = latt_desc[0] * latt_desc[1] * latt_desc[2] * latt_desc[3]
    return operations_per_point * vol

'''
    计算GFlop/s, time: second
'''
def GFlops (flop: int, time: float) :
    return flop / time / 1e9

'''
    验证qcu的正确性，并计算运行时间
'''
def validate_qcu(
        my_m_input, 
        warm_flag = False
    ) :
    
    from pyquda.mpi import comm, rank, size, grid, coord, gpuid
    p_mrhs = [LatticeFermion(latt_size, Nc, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)) for _ in range(my_m_input)]
    # p_mrhs = [LatticeFermion(latt_size, Nc, cp.ones((Lt, Lz, Ly, Lx, Ns, Nc * 2)).view(cp.complex128)) \
    #     for i in range(my_m_input)]

    # allocate memory for quda and qcu
    quda_Mp_mrhs = [LatticeFermion(latt_size, Nc) for _ in range(my_m_input)]
    qcu_Mp_mrhs  = [LatticeFermion(latt_size, Nc) for _ in range(my_m_input)]

    dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False)
    U      = gauge_utils.gaussGauge(latt_size, 0)
    dslash.loadGauge(U)
    cp.cuda.runtime.deviceSynchronize()


    # qcu code 
    qcu.set_tensor_core_flag(0)
    qcu.getDslash(0, mass, 0)           # 参数1：0----WILSON， 参数3暂时未使用
    qcu.loadQcuGauge(U.data_ptr, Precision.kPrecisionDouble)		# 2---double 1--float 0---half

    cp.cuda.runtime.deviceSynchronize()
    qcu_calculate_time = 0
    
    t1 = perf_counter()
    qcu.start_dslash(0, 0)	# param1 : parity  param2: dagger
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_calculate_time += t2 - t1

    t1 = perf_counter()
    qcu.start_dslash(1, 0)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_calculate_time += t2 - t1

    average_difference = cp.sum(cp.array([cp.linalg.norm(quda_Mp_mrhs[i].data - qcu_Mp_mrhs[i].data) / cp.linalg.norm(quda_Mp_mrhs[i].data) for i in range(my_m_input)])) / my_m_input
    print(f'rank {rank}, average difference: , {average_difference}')

    return qcu_calculate_time


def test_dslash(
        color, 
        m_rhs, 
        input_prec, 
        dslash_prec, 
        # quda_time, 
        qcu_time, 
        # qcu_calculate_time,
        warmup_flag = False) :
    
    qcu.initGridSize(grid, param, color, m_rhs, input_prec, dslash_prec)
    
    total_quda_time = 0
    total_qcu_time = 0

    if (not warmup_flag):
        print(f'=========== mrhs = {my_m_input} condition begin ===========')
    # iteration = 1
    # for _ in range(iteration) :
    qcu_calculate_time = validate_qcu(m_rhs)
    
    flop = get_wilson_flop_4dim(latt_size, Nc) * my_m_input
    flops = GFlops(flop, qcu_calculate_time)
    print(f'precision: {Precision.get_precision_name(dslash_prec)}, qcu_calculate_time = {qcu_calculate_time}, flop = {flop}, flops = {flops}')
    if not warmup_flag :
        # quda_time += quda_time
        qcu_time.append(qcu_calculate_time)
        print(f'=========== mrhs = {my_m_input} condition end ===========')

        print(f'operations_per_dslash / time: { max_input * operations_per_dslash / (qcu_calculate_time * 1e-9)}')
    qcu.finalizeQcu()
    cp.cuda.runtime.deviceSynchronize()
    return flops
if __name__ == '__main__' :
    # _ = input()
    max_input = 32
    # my_n_color = Nc

    # operations_per_point = (2 * Nd * Nc * Ns) + (2 * Nd * Ns / 2 * (8 * Nc-2)*Nc) + ((2 * Nd - 1) * 2 * Nc * Ns)
    # operations_per_dslash = operations_per_point * Lx * Ly * Lz * Lt

    my_input_prec  = Precision.kPrecisionDouble
    my_dslash_prec = Precision.kPrecisionSingle

    # quda_time = []
    qcu_time = []
    qcu_calculate_time = []

    # warm up
    # test_dslash(Nc, 1, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
    #     quda_average_time = quda_average_time, qcu_average_time = qcu_average_time, warmup_flag=True)
    # warm up end

    operations_per_point = (2 * Nd * Nc * Ns) + (2 * Nd * Ns / 2 * (8 * Nc-2)*Nc) + ((2 * Nd - 1) * 2 * Nc * Ns)
    operations_per_dslash = operations_per_point * Lx * Ly * Lz * Lt


    flops = []
    for my_m_input in range(max_input, max_input+1):
        single_flops = test_dslash(
            Nc, 
            my_m_input, 
            input_prec=my_input_prec, 
            dslash_prec=my_dslash_prec, 
            # quda_time = quda_time, 
            qcu_time = qcu_time, 
            # qcu_calculate_time = qcu_calculate_time,
            warmup_flag = False
        )
        flops.append(single_flops)
    # print(f'quda_time: {quda_time}')    
    print(f'qcu_time: {qcu_time}')
    # print(f'qcu_calculate_time: {qcu_calculate_time}')
    print(f'flops: {flops}')
    # print(f'operations_per_dslash / time: {operations_per_dslash * max_input / (qcu_calculate_time[0] * 1e-9)}')
    # print(f'qcu_calculate_time = {qcu_calculate_time}')

