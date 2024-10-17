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

Nd, Ns, Nc = 4, 4, 3

# latt_size = [8, 8, 8, 16]
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

def test_sun(round, my_m_input, self_color = 3):
  from pyquda.mpi import comm, rank, size, grid, coord, gpuid

  p_mrhs = [LatticeFermion(latt_size, self_color, cp.random.randn(Lt, Lz, Ly, Lx, Ns, self_color * 2).view(cp.complex128)) \
            for i in range(my_m_input)]

  qcu_Mp_mrhs = [LatticeFermion(latt_size, self_color) for i in range(my_m_input)]

  if self_color == 3:
    U = gauge_utils.gaussGauge(latt_size, 0)
  else:
    U = gauge_utils.unitGauge(latt_size, self_color)
    file_path = 'gaugeSU4_16x16x16x32.bin'
    qcu.read_gauge_from_file(U.data_ptr, file_path.encode('utf-8'))

  #my code 
  qcu.loadQcuGauge(U.data_ptr, 2)		# 2---double 1--float 0---half
  qcu.getDslash(0, mass) # 0----WILSON
  cp.cuda.runtime.deviceSynchronize()

  t1 = perf_counter()
  for i in range(my_m_input):
    qcu.pushBackFermions(qcu_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr)
  qcu.start_dslash(0, 0)	# param1 : parity  param2: dagger

  for i in range(my_m_input):
    qcu.pushBackFermions(qcu_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr)
  qcu.start_dslash(1, 0)
  cp.cuda.runtime.deviceSynchronize()
  t2 = perf_counter()
  qcu_dslash_time = t2 - t1

  print(f"Qcu dslash:  {qcu_dslash_time} sec")
  # print(f"Qcu_mrhs[0] = {qcu_Mp_mrhs[0].data[0, 0, 0, 0, 0, 0]}")
  return qcu_dslash_time


def test_dslash(my_n_color, my_m_input, input_prec, dslash_prec, qcu_average_time, self_color)->int:
  qcu.initGridSize(grid, param, my_n_color, my_m_input, input_prec, dslash_prec)
  
  total_qcu_time = 0

  print(f'=========== mrhs = {my_m_input} condition begin ===========')
  iteration = 10
  for i in range(iteration) :
    qcu_time = test_sun(i, my_m_input, my_n_color)
    total_qcu_time += qcu_time

  qcu_average_time.append(total_qcu_time / iteration)
  print(f'=========== mrhs = {my_m_input} condition end ===========')

  qcu.finalizeQcu()
  cp.cuda.runtime.deviceSynchronize()

if __name__ == '__main__' :
  max_input = 6
 

  my_input_prec  = double_prec
  my_dslash_prec = double_prec


  su3_total_time  = []
  su4_total_time  = []

  my_n_color = 3
  for my_m_input in range(1, max_input+1):
    test_dslash(my_n_color, my_m_input, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
                qcu_average_time = su3_total_time, self_color=my_n_color)
  my_n_color = 4
  for my_m_input in range(1, max_input+1):
    test_dslash(my_n_color, my_m_input, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
                qcu_average_time = su4_total_time, self_color=my_n_color)


  # print(f'qcu su3 total time: {su3_total_time}')
  print(f'qcu su4 total time: {su4_total_time}')
  x = np.arange(1, max_input+1, 1)
  su3_dslash_per_rhs  = su3_total_time / x
  su4_dslash_per_rhs  = su4_total_time / x

  # marker: s p ^  h H
  plt.plot(x, su3_total_time, label='SU(3)', marker='s')
  plt.plot(x, su4_total_time, linestyle = '--',label='SU(4)', marker='o')

  plt.title(f'mrhs dslash total time, latt size = {latt_size}, prec = {precision_table[my_dslash_prec]}')
  plt.xlabel('m_input')
  plt.ylabel('time')
  plt.legend()
  plt.show()
  plt.savefig('mrhs_dslash_total.png')

  plt.clf()
  # plt.plot(x, quda_average_time, label='quda', marker = 'o')
  plt.plot(x, su3_dslash_per_rhs, label='SU(3)', marker='s')
  plt.plot(x, su4_dslash_per_rhs, linestyle = '--',label='SU(4)', marker='o')
  plt.title(f'mrhs dslash per mrhs, latt size = {latt_size}, prec = {precision_table[my_dslash_prec]}')
  plt.xlabel('m_input')
  plt.ylabel('time')
  plt.legend()
  plt.show()
  plt.savefig('mrhs_dslash_average.png')