# this file is modified from test.dslash.qcu.py
# 
import copy
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

# latt_size = [8,8,8,8]
latt_size = [16, 16, 16, 16]

grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
print(f'vol = {Vol}')
xi_0, nu = 1, 1
# mass=-3.5
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
max_iteration = 100
max_prec = 1e-6

Nc = 4
def draw_table_mrhs (x, y1, y1_label, y2, y2_label, table_name, my_dslash_prec) :
  plt.clf()
  plt.plot(x, y1, label=y1_label, marker = 'o')
  plt.plot(x, y2, linestyle = '--', label=y2_label, marker='o')
  plt.title(f'{table_name}, latt size = {latt_size}, prec = {precision_table[my_dslash_prec]}')
  plt.xlabel('m_input')
  plt.ylabel('time')
  plt.legend()
  plt.show()
  plt.savefig(table_name)


def test_mpi(my_m_input, warm_flag = False):
  from pyquda.mpi import rank
  from pyquda.enum_quda import QudaInverterType
 
#   def matQcu(qcu_Mp_mrhs, p_mrhs) :
#     for i in range(my_m_input):
#         qcu.pushBackFermions(qcu_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr)
#     qcu.begin_gather()
#     cp.cuda.runtime.deviceSynchronize()

#     qcu.start_dslash(0, 0)
#     cp.cuda.runtime.deviceSynchronize()

#     qcu.begin_scatter()
#     cp.cuda.runtime.deviceSynchronize()

#     for i in range(my_m_input):
#         qcu.pushBackFermions(qcu_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr)
#     qcu.begin_gather()
#     cp.cuda.runtime.deviceSynchronize()

#     qcu.start_dslash(1, 0)
#     cp.cuda.runtime.deviceSynchronize()

#     qcu.begin_scatter()
#     cp.cuda.runtime.deviceSynchronize()

#     kappa = 1.0 / (2.0 * (4.0 + mass))
#     for i in range(my_m_input):
#         qcu_Mp_mrhs[i].data -= kappa * p_mrhs[i].data

  # get quda_dslash operator
  quda_dslash = core.getDslash(latt_size, mass, max_prec, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False)
  quda_dslash.invert_param.inv_type = QudaInverterType.QUDA_BICGSTAB_INVERTER
  # quda_dslash.invert_param.chrono_precision = max_prec
  
  U = gauge_utils.gaussGauge(latt_size, 0)
#   U = gauge_utils.unitGauge(latt_size, Nc)
  print(U.data.shape)

  print('==============BEGIN=================')
  # allocate my_m_input number of fermions
  x_mrhs = [LatticeFermion(latt_size, Nc, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)) \
            for _ in range(my_m_input)]
#   b_mrhs = [LatticeFermion(latt_size, Nc) for _ in range(my_m_input)]
#   for i in range(my_m_input):
#     quda.MatQuda(b_mrhs[i].even_ptr, x_mrhs[i].even_ptr, quda_dslash.invert_param)

  # qcu_result
  qcu_x_mrhs  = [LatticeFermion(latt_size, Nc) for _ in range(my_m_input)]
  qcu_b_mrhs  = [LatticeFermion(latt_size, Nc) for _ in range(my_m_input)]
  b_mrhs = [LatticeFermion(latt_size, Nc) for _ in range(my_m_input)]
#   quda_dslash.loadGauge(U)
  cp.cuda.runtime.deviceSynchronize()

  # quda invert
#   t1 = perf_counter()
#   for i in range(my_m_input):
#     quda.invertQuda(quda_x_mrhs[i].data_ptr, b_mrhs[i].data_ptr, quda_dslash.invert_param)
#   cp.cuda.runtime.deviceSynchronize()
#   t2 = perf_counter()
  quda_inverter_time = 0

  # check if quda is correct
  # diff_b_b_groundtruth = 0

#   for i in range(my_m_input):
#     qcu.qcu.mat_Qcu(daggerFlag)
#     quda.MatQuda(quda_b_mrhs[i].even_ptr, quda_x_mrhs[i].even_ptr, quda_dslash.invert_param)
#     print(f'rank {rank}, rhs {i} difference between quda_x_mrhs and x_mrhs: \
#           , {cp.linalg.norm(b_mrhs[i].data - quda_b_mrhs[i].data) / cp.linalg.norm(b_mrhs[i].data)}')
#     # diff_b_b_groundtruth += cp.linalg.norm(quda_x_mrhs[i].data - x_mrhs[i].data) / cp.linalg.norm(x_mrhs[i].data)
#   # print(f'rank {rank}, average difference between quda_x_mrhs and x_mrhs: , {diff_b_b_groundtruth / my_m_input}')

  #my code 
  qcu.set_tensor_core_flag(0)
  qcu.set_residual_combine_flag(0)
  qcu.getDslash(0, mass, 0) # 0----WILSON
#   qcu.loadQcuGauge(U.data_ptr, 2)		# 2---double 1--float 0---half

  qcu.read_gauge_from_file(U.data_ptr, 'test_gauge/test_gauge_su4.hdf5'.encode('utf-8'))
#   print(f'U[0] = {U.data[0, 0, 0, 0, 0, 0]}')
#   print(f'det(U[0]) = {cp.linalg.det(U.data[0, 0, 0, 0, 0, 0])}')
  qcu.loadQcuGauge(U.data_ptr, 2)

  for i in range (my_m_input):
    qcu.pushBackFermions(b_mrhs[i].data_ptr, x_mrhs[i].data_ptr)
  qcu.mat_Qcu(0)
#   matQcu(qcu_b_mrhs, qcu_x_mrhs)
  cp.cuda.runtime.deviceSynchronize()

  cp.cuda.runtime.deviceSynchronize()
  # qcu invert 
  t1 = perf_counter()
  for i in range(my_m_input):
    qcu.pushBackFermions(qcu_x_mrhs[i].data_ptr, b_mrhs[i].data_ptr)
  qcu.qcuInvert(max_iteration, max_prec)
  cp.cuda.runtime.deviceSynchronize()
  t2 = perf_counter()
  qcu_inverter_time = t2 - t1

  if (not warm_flag):
    print(f'Quda solver: {quda_inverter_time}sec')
    print(f'Qcu dslash:  {qcu_inverter_time} sec')

  cp.cuda.runtime.deviceSynchronize()
  print(f'rank {rank}, input============================================')
  # for i in range(my_m_input):
  #   print(f'b[i][0] = {b_mrhs[i].data[0, 0, 0, 0, 0]}')
  # print(f'=============================================================')
  cp.cuda.runtime.deviceSynchronize()
  diff_b_b_groundtruth = 0
  
#   for i in range(my_m_input):
#     # quda.MatQuda(qcu_b_mrhs[i].even_ptr, qcu_x_mrhs[i].even_ptr, quda_dslash.invert_param)
#     print(f'rank {rank}, rhs {i} difference between qcu_x_mrhs and x_mrhs: \
#           , {cp.linalg.norm(qcu_x_mrhs[i].data - x_mrhs[i].data) / cp.linalg.norm(x_mrhs[i].data)}')
#   cp.cuda.runtime.deviceSynchronize()
  for i in range (my_m_input):
    qcu.pushBackFermions(qcu_b_mrhs[i].data_ptr, qcu_x_mrhs[i].data_ptr)
  qcu.mat_Qcu(0)
  for i in range(my_m_input):
    print(f'rank {rank}, rhs {i} difference between qcu_x_mrhs and x_mrhs: \
          , {cp.linalg.norm(b_mrhs[i].data - qcu_b_mrhs[i].data) / cp.linalg.norm(b_mrhs[i].data)}')


  print('==============END=================')
  return quda_inverter_time, qcu_inverter_time


def test_bicgstab(my_n_color, my_m_input, input_prec, dslash_prec, quda_average_time, qcu_average_time, warmup_flag = False)->int:
  qcu.initGridSize(grid, param, my_n_color, my_m_input, input_prec, dslash_prec)
  
  total_quda_time = 0
  total_qcu_time = 0

  # if (not warmup_flag):
  #   print(f'=========== mrhs = {my_m_input} condition begin ===========')
  print(f'=========== mrhs = {my_m_input} condition begin ===========')

  iteration = 1
  for _ in range(iteration) :
    quda_time, qcu_time = test_mpi(my_m_input)
    total_quda_time += quda_time
    total_qcu_time += qcu_time
  
  if (not warmup_flag):
    quda_average_time.append(total_quda_time / iteration)
    qcu_average_time.append(total_qcu_time / iteration)
    print(f'=========== mrhs = {my_m_input} condition end ===========')

  qcu.finalizeQcu()
  cp.cuda.runtime.deviceSynchronize()



if __name__ == '__main__' :
  max_input = 16
  my_n_color = Nc

  my_input_prec  = double_prec
  my_dslash_prec = half_prec

  quda_total_time = []
  qcu_total_time  = []

  # warm up
  # test_bicgstab(my_n_color, 1, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
  #   quda_average_time = quda_total_time, qcu_average_time = qcu_total_time, warmup_flag=True)
  # warm up end

  # for my_m_input in range(1, max_input+1):
  input = 8
  for _ in range(4):
    # print(f'=========== mrhs = {my_m_input} condition begin ===========')
    print(f'BEGIN PROGRAM=======cur input = {input} ===>>>>>>>>>>>>')
    test_bicgstab(my_n_color, my_m_input=input, input_prec=my_input_prec, dslash_prec=my_dslash_prec, quda_average_time = quda_total_time, qcu_average_time = qcu_total_time)
    print(f'quda_mrhs_total_time: {quda_total_time}')
    print(f'qcu_mrhs_total_time: {qcu_total_time}')
    cp.cuda.runtime.deviceSynchronize()
    print(f'=========== mrhs = {input} condition end ===========')

#   x = np.arange(1, max_input+1, 1)
#   quda_per_rhs = quda_total_time / x
#   qcu_per_rhs  = qcu_total_time / x

#   draw_table_mrhs(x, quda_per_rhs, 'quda', qcu_per_rhs, 'qcu', 'bicgstab_result_total_time.png', my_dslash_prec)
#   draw_table_mrhs(x, quda_per_rhs, 'quda', qcu_per_rhs, 'qcu', 'bicgstab_result_average_time_per_rhs.png', my_dslash_prec)
