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

Nd, Ns, Nc = 4, 4, 3

latt_size = [8,8,8,8]
# latt_size = [16, 16, 16, 16]

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
  # get quda_dslash operator
  quda_dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False)
  U = gauge_utils.gaussGauge(latt_size, 0)

  print('==============BEGIN=================')
  # allocate my_m_input number of fermions
  # x_mrhs = [LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128)) \
  #           for _ in range(my_m_input)]

  x_mrhs = [LatticeFermion(latt_size, cp.ones((Lt, Lz, Ly, Lx, Ns, Nc * 2)).view(cp.complex128)) \
            for _ in range(my_m_input)]
  b_mrhs = [LatticeFermion(latt_size) for _ in range(my_m_input)]
  for i in range(my_m_input):
    quda.MatQuda(b_mrhs[i].even_ptr, x_mrhs[i].even_ptr, quda_dslash.invert_param)

  # quda_result
  quda_x_mrhs = [LatticeFermion(latt_size) for _ in range(my_m_input)]
  # qcu_result
  qcu_x_mrhs  = [LatticeFermion(latt_size) for _ in range(my_m_input)]



  quda_dslash.loadGauge(U)
  cp.cuda.runtime.deviceSynchronize()

  # quda invert
  t1 = perf_counter()
  for i in range(my_m_input):
    quda.invertQuda(quda_x_mrhs[i].data_ptr, b_mrhs[i].data_ptr, quda_dslash.invert_param)
  cp.cuda.runtime.deviceSynchronize()
  t2 = perf_counter()
  quda_inverter_time = t2 - t1

  # print('debug info, new b')
  # test 
  # new b = b_{o} + kappa D_{oe} b_{e}
  b = LatticeFermion(latt_size)
  temp = LatticeFermion(latt_size)
  quda.dslashQuda(b.odd_ptr, b_mrhs[0].even_ptr, quda_dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
  new_b = copy.copy(b_mrhs[0].data[1] + 0.125 * b.data[1])

  r0 = copy.copy(new_b)

  b.data[0][:] = r0
  r0_norm = cp.linalg.norm(r0)
  r1      = copy.copy(r0)
  p       = copy.copy(r0)
  x_old   = cp.zeros_like(r0)

  for i in range (max_iteration) :
    # rho_j = <r_i , r0'>
    rho_j = cp.dot(r0.reshape(1, -1).conj(), r1.reshape(-1, 1))
    # v_i = A p = kappa^2 D_{oe} D_{eo} p
    b.data[0][:] = p
    quda.dslashQuda(temp.even_ptr, b.even_ptr, quda_dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(b.odd_ptr, temp.even_ptr, quda_dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    b.data[1][:] = b.data[0] - 0.125 * 0.125 * b.data[1]
    vi = copy.copy(b.data[1])
    vi_r0_prod = cp.dot(r0.reshape(1, -1).conj(), vi.reshape(-1, 1))
    alpha = rho_j / vi_r0_prod
    s_j = r1 - alpha * vi

    # t = s_j kappa^2 D_{oe} D_{eo} s_j
    b.data[0][:] = s_j
    quda.dslashQuda(temp.even_ptr, b.even_ptr, quda_dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(b.odd_ptr, temp.even_ptr, quda_dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    b.data[1][:] = b.data[0] - 0.125 * 0.125 * b.data[1]
    t = copy.copy(b.data[1])
  
    t_sj_prod = cp.dot(t.reshape(1, -1).conj(), s_j.reshape(-1, 1))
    t_t_prod = cp.dot(t.reshape(1, -1).conj(), t.reshape(-1, 1))
    omega = t_sj_prod / t_t_prod
    # print(f'omega = {omega}, t_sj_prod = {t_sj_prod}, t_t_prod = {t_t_prod}, norm <s, s> = {cp.linalg.norm(s_j)}')
    
    x_old[:] = x_old + alpha * p + omega * s_j
    r1[:] = s_j - omega * t

    print(f'iteration {i}, norm(r1) = {cp.linalg.norm(r1)}, norm_b = {r0_norm}, diff = {cp.linalg.norm(r1) / r0_norm}')
    if (cp.linalg.norm(r1) / r0_norm < max_prec):
      print(f'converged at iteration {i}')
      print(f'my res = {x_old[0, 0, 0, 0]}')
      break
    rho_j1 = cp.dot(r0.reshape(1, -1).conj(), r1.reshape(-1, 1))
    beta = (alpha / omega) * (rho_j1 / rho_j)
    ppp = p - omega * vi
    p_new = r1 + beta * ppp

    p[:] = p_new
    # print(f'round {i}, alpha = {alpha}, omega = {omega}, beta = {beta}')

    # print(f'p_new = {p_new[0, 0, 0, 0]}')
  b.data[1][:] = x_old
  # D_{eo} x_o
  quda.dslashQuda(temp.even_ptr, b.odd_ptr, quda_dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
  temp_res = copy.copy(temp.data[0])
  # print(f'temp_res = {temp_res[0, 0, 0, 0]}')
  # even res : b_{e} + kappa D_{oe} x_o
  origin_be = copy.copy(b_mrhs[0].data[0])
  even_res = origin_be + 0.125 * temp_res
  print(f'even_res = {even_res[0, 0, 0, 0]}')

  # check if quda is correct
  quda_b_mrhs = [LatticeFermion(latt_size) for _ in range(my_m_input)]
  diff_b_b_groundtruth = 0
  for i in range(my_m_input):
    quda.MatQuda(quda_b_mrhs[i].data_ptr, quda_x_mrhs[i].data_ptr, quda_dslash.invert_param)
    diff_b_b_groundtruth += cp.linalg.norm(quda_b_mrhs[i].data - b_mrhs[i].data) / cp.linalg.norm(b_mrhs[i].data)
  print(f'rank {rank}, average difference between quda_b_mrhs and b_mrhs: , {diff_b_b_groundtruth / my_m_input}')

  #my code 
  qcu.loadQcuGauge(U.data_ptr, 2)		# 2---double 1--float 0---half
  qcu.getDslash(0, mass) # 0----WILSON
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

  # check the result
  print(f'quda_x_even = {quda_x_mrhs[0].data[0, 0, 0, 0, 0]}')
  print(f'quda_x_odd = {quda_x_mrhs[0].data[1, 0, 0, 0, 0]}')
  print(f'qcu_x_even = {qcu_x_mrhs[0].data[0, 0, 0, 0, 0]}')
  print(f'qcu_x_odd = {qcu_x_mrhs[0].data[1, 0, 0, 0, 0]}')
  print(f'groundtruth = {x_mrhs[0].data[0, 0, 0, 0, 0]}')

  average_difference = cp.sum(cp.array([cp.linalg.norm(quda_x_mrhs[i].data - qcu_x_mrhs[i].data) / cp.linalg.norm(quda_x_mrhs[i].data) \
              for i in range(my_m_input)])) / my_m_input
  print(f'rank {rank}, average difference: , {average_difference}')
  print('==============END=================')
  return quda_inverter_time, qcu_inverter_time


def test_dslash(my_n_color, my_m_input, input_prec, dslash_prec, quda_average_time, qcu_average_time, warmup_flag = False)->int:
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
  max_input = 1
  my_n_color = 3

  my_input_prec  = double_prec
  my_dslash_prec = double_prec

  quda_total_time = []
  qcu_total_time  = []

  # warm up
  # test_dslash(my_n_color, 1, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
  #   quda_average_time = quda_total_time, qcu_average_time = qcu_total_time, warmup_flag=True)
  # warm up end

  my_m_input = 1
  # for my_m_input in range(1, max_input+1):
    # print(f'=========== mrhs = {my_m_input} condition begin ===========')
  print(f'BEGIN PROGRAM=================>>>>>>>>>>>>')
  test_dslash(my_n_color, my_m_input, input_prec=my_input_prec, dslash_prec=my_dslash_prec, quda_average_time = quda_total_time, qcu_average_time = qcu_total_time)
  
  print(f'quda_mrhs_total_time: {quda_total_time}')
  print(f'qcu_mrhs_total_time: {qcu_total_time}')
  
  x = np.arange(1, max_input+1, 1)
  quda_per_rhs = quda_total_time / x
  qcu_per_rhs  = qcu_total_time / x

  draw_table_mrhs(x, quda_per_rhs, 'quda', qcu_per_rhs, 'qcu', 'dslash_result_total_time.png', my_dslash_prec)
  draw_table_mrhs(x, quda_per_rhs, 'quda', qcu_per_rhs, 'qcu', 'dslash_result_average_time_per_rhs.png', my_dslash_prec)