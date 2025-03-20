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

class Precision(Enum):
    HALF = 0
    FLOAT = 1 
    DOUBLE = 2
    
    @property
    def name(self):
        return self._name_.lower()

# 使用示例:
# my_input_prec = Precision.DOUBLE
# print(my_input_prec.value)  # 输出: 2
# print(my_input_prec.name)   # 输出: "double"

class Config:
    def __init__(self):
        self.Nd, self.Ns, self.Nc = 4, 4, 3
        self.latt_size = [8, 8,16, 32]
        self.grid_size = [1, 1, 1, 1]
        self.xi_0, self.nu = 1, 1
        self.mass = 0
        self.coeff_r, self.coeff_t = 0, 0  # wilson

config = Config()

mpi.init(config.grid_size)

param = qcu.QcuParam()
grid = qcu.QcuGrid()
param.lattice_size = config.latt_size
grid.grid_size = config.grid_size


class TestDslash:
    def __init__(self, config):
        self.config = config
        self.setup_lattice()
        
    def setup_lattice(self):
        Lx, Ly, Lz, Lt = self.config.latt_size

def test_mpi(round, my_m_input, warm_flag = False):
    Lx, Ly, Lz, Lt = config.latt_size
    from pyquda.mpi import comm, rank, size, grid, coord, gpuid
    p_mrhs = [LatticeFermion(config.latt_size, 3, cp.random.randn(Lt, Lz, Ly, Lx, config.Ns, config.Nc * 2).view(cp.complex128)) \
            for i in range(my_m_input)]

    quda_Mp_mrhs = [LatticeFermion(config.latt_size, 3) for i in range(my_m_input)]
    qcu_Mp_mrhs = [LatticeFermion(config.latt_size, 3) for i in range(my_m_input)]

    quda_dslash = core.getDslash(config.latt_size, config.mass, 1e-9, 1000, config.xi_0, config.nu, config.coeff_t, config.coeff_r, multigrid=False, anti_periodic_t=False)
    U = gauge_utils.gaussGauge(config.latt_size, 0)

    quda_dslash.loadGauge(U)
    cp.cuda.runtime.deviceSynchronize()

    t1 = perf_counter()
    for i in range(my_m_input):
        quda.MatQuda(quda_Mp_mrhs[i].even_ptr, p_mrhs[i].even_ptr, quda_dslash.invert_param)
    # quda.dslashQuda(quda_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr, quda_dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    quda_dslash_time = t2 - t1

    #my code 
    qcu.loadQcuGauge(U.data_ptr, 2)		# 2---double 1--float 0---half
    qcu.getDslash(0, config.mass) # 0----WILSON
    cp.cuda.runtime.deviceSynchronize()

    t1 = perf_counter()
    for i in range(my_m_input):
        qcu.pushBackFermions(qcu_Mp_mrhs[i].even_ptr, p_mrhs[i].even_ptr)
    qcu.mat_Qcu(0)	# param: dagger
    cp.cuda.runtime.deviceSynchronize()
    print (f'qcu[0, 0, 0, 0, 0] = {qcu_Mp_mrhs[0].data[0, 0, 0, 0, 0]}')
    t2 = perf_counter()
    qcu_dslash_time = t2 - t1

    # if (not warm_flag):
    print(f"Quda dslash: {quda_dslash_time}sec \nQcu dslash:  {qcu_dslash_time} sec")

    average_difference = cp.sum(cp.array([cp.linalg.norm(quda_Mp_mrhs[i].data - qcu_Mp_mrhs[i].data) / cp.linalg.norm(quda_Mp_mrhs[i].data) \
            for i in range(my_m_input)])) / my_m_input
    print(f'rank {rank}, average difference: , {average_difference}')
    print('===============================')
    return quda_dslash_time, qcu_dslash_time



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
    max_input = 1
    my_n_color = config.Nc

    my_input_prec  = Precision.DOUBLE
    my_dslash_prec = Precision.DOUBLE
    print(my_input_prec.value)

    quda_average_time = []
    qcu_average_time  = []


    # # warm up
    # test_dslash(my_n_color, 1, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
    #     quda_average_time = quda_average_time, qcu_average_time = qcu_average_time, warmup_flag=True)
    # # warm up end

    for my_m_input in range(1, max_input+1):
        test_dslash(my_n_color, my_m_input, input_prec=my_input_prec.value, dslash_prec=my_dslash_prec.value, quda_average_time = quda_average_time, qcu_average_time = qcu_average_time)
    
    print(f'quda_average_time: {quda_average_time}')
    print(f'qcu_average_time: {qcu_average_time}')
    
    x = np.arange(1, max_input+1, 1)
    quda_per_rhs = quda_average_time / x
    qcu_per_rhs  = qcu_average_time / x

    # plt.plot(x, quda_per_rhs, label='quda', marker = 'o')
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