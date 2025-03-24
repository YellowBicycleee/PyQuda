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
Nc = 3
latt_size = [4, 4, 4, 4] # lattice description
# latt_size = [16, 16, 16, 16] # lattice description
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
                        if point_diff < 1e-3:
                            same.append(point_diff)
                        else:
                            diff.append(point_diff)
    print(f'same: {len(same)}, diff: {len(diff)}, total: {len(same) + len(diff)}')
    print(f'first element : {cp.linalg.norm(quda_result.data[0, 0, 0, 0, 0] - qcu_result.data[0, 0, 0, 0, 0]) / cp.linalg.norm(quda_result.data[0, 0, 0, 0, 0])}')
    print(f'quda = {quda_result.data[0, 0, 0, 0, 0]}')
    print(f'qcu  = {qcu_result.data[0, 0, 0, 0, 0]}')
    print(f'quda-qcu element : \n{quda_result.data[0, 0, 0, 0, 0] - qcu_result.data[0, 0, 0, 0, 0]}')


def test_mpi(round, my_m_input, warm_flag = False):
    from pyquda.mpi import comm, rank, size, grid, coord, gpuid
    p_mrhs = [LatticeFermion(latt_size, Nc, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)) \
            for i in range(my_m_input)]
    # p_mrhs = [LatticeFermion(latt_size, Nc, cp.ones((Lt, Lz, Ly, Lx, Ns, Nc * 2)).view(cp.complex128)) \
    #     for i in range(my_m_input)]

    quda_Mp_mrhs = [LatticeFermion(latt_size, Nc) for i in range(my_m_input)]
    qcu_Mp_mrhs = [LatticeFermion(latt_size, Nc) for i in range(my_m_input)]

    dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False)
    U = gauge_utils.gaussGauge(latt_size, 0)


    dslash.loadGauge(U)
    cp.cuda.runtime.deviceSynchronize()
    # print(f'Gauge[0] = \n{U.data[0, 0, 0, 0, 0, 0]}')
    # print(f'fermion input[0] = \n{p_mrhs[0].data[1, 0, 0, 0, 0]}')

    t1 = perf_counter()
    for i in range(my_m_input):
        quda.dslashQuda(quda_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
        quda.dslashQuda(quda_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    quda_dslash_time = t2 - t1

    #my code 
    qcu.set_tensor_core_flag(1)
    qcu.getDslash(0, mass, 0) # 0----WILSON, 关闭反周期
    qcu.loadQcuGauge(U.data_ptr, 2)		# 2---double 1--float 0---half


    cp.cuda.runtime.deviceSynchronize()

    calculate_time = 0
    scatter_time = 0
    gather_time = 0
    
    for i in range(my_m_input):
        qcu.pushBackFermions(qcu_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr)
    t1 = perf_counter()
    qcu.begin_gather()
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    gather_time += t2 - t1
    
    t1 = perf_counter()
    qcu.start_dslash(0, 0)	# param1 : parity  param2: dagger
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    calculate_time += t2 - t1

    t1 = perf_counter()
    qcu.begin_scatter()
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    scatter_time += t2 - t1


    for i in range(my_m_input):
        qcu.pushBackFermions(qcu_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr)
    t1 = perf_counter()
    qcu.begin_gather()
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    gather_time += t2 - t1

    t1 = perf_counter()
    qcu.start_dslash(1, 0)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    calculate_time += t2 - t1

    t1 = perf_counter()
    qcu.begin_scatter()
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    scatter_time += t2 - t1
    # for i in range(my_m_input):
    #     print(f'fermion[{i}, 0, 0, 0, Lx // 2 -1,0 ] = \n{p_mrhs[i].data[1, 0, 0, 0, 0, 0] - 1j * p_mrhs[i].data[1, 0, 0, 0, 0, 3]}')
    #     print(f'fermion[{i}, 0, 0, 0, Lx // 2 -1,1 ] = \n{p_mrhs[i].data[1, 0, 0, 0, 0, 1] - 1j * p_mrhs[i].data[1, 0, 0, 0, 0, 2]}')

    if (not warm_flag):
        print(f"Quda dslash: {quda_dslash_time}sec \n"
              f"Qcu dslash: total {calculate_time + scatter_time + gather_time}sec, calculate {calculate_time}, scatter {scatter_time}, gather {gather_time}")
    # for i in range(my_m_input) :
    #   print(f'rank {rank}, Mp[{i}] difference: , \
    
    #         {cp.linalg.norm(quda_Mp_mrhs[i].data - qcu_Mp_mrhs[i].data) / cp.linalg.norm(quda_Mp_mrhs[i].data)}')
    # for i in range(my_m_input):
    #     if mpi.rank == 0:
    #         print(f'quda_res = \n{quda_Mp_mrhs[i].data[1, Lt - 1, Lz - 1, Ly - 1, Lx // 2 - 1]}')
    #         print(f'qcu_res = \n{qcu_Mp_mrhs[i].data[1, Lt - 1, Lz - 1, Ly - 1, Lx // 2 - 1]}')
    average_difference = cp.sum(cp.array([cp.linalg.norm(quda_Mp_mrhs[i].data - qcu_Mp_mrhs[i].data) / cp.linalg.norm(quda_Mp_mrhs[i].data) \
                for i in range(my_m_input)])) / my_m_input
    # if 
    # assert(average_difference < 1e-13)
    print(f'rank {rank}, average difference: , {average_difference}')

    # average_difference = cp.sum(cp.array([cp.linalg.norm(quda_Mp_mrhs[i].data[0] - qcu_Mp_mrhs[i].data[0]) / cp.linalg.norm(quda_Mp_mrhs[i].data[0]) \
    #             for i in range(my_m_input)])) / my_m_input
    # print(f'rank {rank}, even average difference: , {average_difference}')
    # average_difference = cp.sum(cp.array([cp.linalg.norm(quda_Mp_mrhs[i].data[1] - qcu_Mp_mrhs[i].data[1]) / cp.linalg.norm(quda_Mp_mrhs[i].data[1]) \
    #             for i in range(my_m_input)])) / my_m_input
    # print(f'rank {rank}, odd average difference: , {average_difference}')
    # for i in range(my_m_input):
    #     print(f"No.{i} result:")
    #     compare_result(quda_Mp_mrhs[i], qcu_Mp_mrhs[i])

    return quda_dslash_time, calculate_time


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
    max_input = 16
    # my_n_color = Nc

    operations_per_point = (2 * Nd * Nc * Ns) + (2 * Nd * Ns / 2 * (8 * Nc-2)*Nc) + ((2 * Nd - 1) * 2 * Nc * Ns)
    operations_per_dslash = operations_per_point * Lx * Ly * Lz * Lt

    my_input_prec  = double_prec
    my_dslash_prec = half_prec

    quda_average_time = []
    qcu_average_time  = []

    # warm up
    # test_dslash(Nc, 1, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
    #     quda_average_time = quda_average_time, qcu_average_time = qcu_average_time, warmup_flag=True)
    # warm up end

    for my_m_input in range(1, max_input+1):
        test_dslash(Nc, my_m_input, input_prec=my_input_prec, dslash_prec=my_dslash_prec, quda_average_time = quda_average_time, qcu_average_time = qcu_average_time)
    
    # print(f'quda_average_time: {np.array(quda_average_time).reshape(-1, 8)}')
    # print(f'qcu_average_time: {np.array(qcu_average_time).reshape(-1, 8)}')
    # x = np.arange(1, max_input+1, 1)
    # quda_per_rhs = quda_average_time / x
    # qcu_per_rhs  = qcu_average_time / x

    # quda_gflops = operations_per_dslash / np.array(quda_per_rhs) * 1e-9
    # qcu_gflops = operations_per_dslash / np.array(qcu_per_rhs) * 1e-9
    # print(f'quda_gflops: {quda_gflops}')
    # print(f'qcu_gflops: {qcu_gflops}')

    # plt.plot(x, quda_per_rhs, label='quda', marker = 's')
    # plt.plot(x, qcu_per_rhs, linestyle = '--',label='qcu', marker='o')
    # plt.title(f'average dslash time per rhs, latt size = {latt_size}, prec = {precision_table[my_dslash_prec]}')
    # plt.xlabel('m_input')
    # plt.ylabel('time')
    # plt.legend()
    # plt.show()
    # plt.savefig('dslash_result_per_rhs.png')

    # plt.clf()
    # plt.plot(x, quda_average_time, label='quda', marker = 'o')
    # plt.plot(x, qcu_average_time, linestyle = '--',label='qcu', marker='o')
    # plt.title(f'average dslash time, latt size = {latt_size}, prec = {precision_table[my_dslash_prec]}')
    # plt.xlabel('m_input')
    # plt.ylabel('time')
    # plt.legend()
    # plt.show()
    # plt.savefig('dslash_result.png')
