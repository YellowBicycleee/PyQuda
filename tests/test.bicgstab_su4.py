# this file is modified from test.dslash.qcu.py
# 
import os
import sys
from time import perf_counter

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

test_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, os.path.join(test_dir, ".."))

from pyquda import init, core, quda, mpi, pyqcu as qcu
from pyquda.enum_quda import QudaParity
from pyquda.field import LatticeFermion
from pyquda.utils import gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nd, Ns = 4, 4

# latt_size = [8, 8,16, 32]
latt_size = [16, 16, 16, 32]

grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
print(f'vol = {Vol}')
xi_0, nu = 1, 1
mass=-3.5
mass=0
#coeff_r, coeff_t = 1,1
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
max_iteration = 1000
max_prec = 1e-9

Nc = 4

def test_mpi(round, my_m_input, warm_flag = False):
  from pyquda.mpi import comm, rank, size, grid, coord, gpuid
  x_mrhs = [LatticeFermion(latt_size, Nc, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)) \
            for i in range(my_m_input)]

  qcu_x_mrhs = [LatticeFermion(latt_size, Nc) for i in range(my_m_input)]
  qcu_b_mrhs = [LatticeFermion(latt_size, Nc) for i in range(my_m_input)]

  if Nc == 3:
    U = gauge_utils.gaussGauge(latt_size, 0)
  else :
    U = gauge_utils.unitGauge(latt_size, Nc)
    file_path = 'gaugeSU4_16x16x16x32.bin'
    qcu.read_gauge_from_file(U.data_ptr, file_path.encode('utf-8'))

  #my code 
  qcu.loadQcuGauge(U.data_ptr, 2)		# 2---double 1--float 0---half
  qcu.getDslash(0, mass) # 0----WILSON
  cp.cuda.runtime.deviceSynchronize()

  t1 = perf_counter()
  for i in range(my_m_input):
    qcu.pushBackFermions(qcu_b_mrhs[i].even_ptr, x_mrhs[i].even_ptr)
  qcu.mat_Qcu(0)	# param: dagger
  cp.cuda.runtime.deviceSynchronize()
  t2 = perf_counter()
  qcu_dslash_time = t2 - t1

  # qcu invert 
  t1 = perf_counter()
  for i in range(my_m_input):
    qcu.pushBackFermions(qcu_x_mrhs[i].data_ptr, qcu_b_mrhs[i].data_ptr)
  qcu.qcuInvert(max_iteration, max_prec)
  cp.cuda.runtime.deviceSynchronize()
  t2 = perf_counter()
  qcu_inverter_time = t2 - t1

  for i in range(my_m_input):
    print(f'rank {rank}, rhs {i} difference between qcu_x_mrhs and x_mrhs: \
          , {cp.linalg.norm(x_mrhs[i].data - qcu_x_mrhs[i].data) / cp.linalg.norm(x_mrhs[i].data)}')
  
  print('===============================')
  return qcu_dslash_time



def test_dslash(my_n_color, my_m_input, input_prec, dslash_prec, quda_average_time, qcu_average_time, warmup_flag = False)->int:
  qcu.initGridSize(grid, param, my_n_color, my_m_input, input_prec, dslash_prec)
  
  total_qcu_time = 0

  if (not warmup_flag):
    print(f'=========== mrhs = {my_m_input} condition begin ===========')
  iteration = 1
  for i in range(iteration) :
    qcu_time = test_mpi(i, my_m_input)
    total_qcu_time += qcu_time
  
  if (not warmup_flag):
    qcu_average_time.append(total_qcu_time / iteration)
    print(f'=========== mrhs = {my_m_input} condition end ===========')

  qcu.finalizeQcu()
  cp.cuda.runtime.deviceSynchronize()

if __name__ == '__main__' :
  max_input = 8
  my_n_color = Nc

  my_input_prec  = double_prec
  my_dslash_prec = double_prec

  quda_average_time = []
  qcu_average_time  = []

  
  # warm up
  # test_dslash(my_n_color, 1, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
    # quda_average_time = quda_average_time, qcu_average_time = qcu_average_time, warmup_flag=True)
  # warm up end

  # for my_m_input in range(1, max_input+1):
    # test_dslash(my_n_color, my_m_input, input_prec=my_input_prec, dslash_prec=my_dslash_prec, quda_average_time = quda_average_time, qcu_average_time = qcu_average_time)
  test_dslash(my_n_color, 1, input_prec=my_input_prec, dslash_prec=my_dslash_prec, quda_average_time = quda_average_time, qcu_average_time = qcu_average_time)

  print(f'quda_average_time: {quda_average_time}')
  print(f'qcu_average_time: {qcu_average_time}')
  
  x = np.arange(1, max_input+1, 1)
  # quda_per_rhs = quda_average_time / x
  qcu_per_rhs  = qcu_average_time / x