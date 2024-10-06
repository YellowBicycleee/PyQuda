# this file is modified from test.dslash.qcu.py
# 
import os
import sys
from time import perf_counter
import copy
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
max_prec = 1e-9
precision_table = ['half', 'float', 'double']

def test_precondition():
  from pyquda.mpi import rank
  from pyquda.enum_quda import QudaInverterType
  # get quda_dslash operator
  quda_dslash = core.getDslash(latt_size, mass, max_prec, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False)
  quda_dslash.invert_param.inv_type = QudaInverterType.QUDA_BICGSTAB_INVERTER
  # quda_dslash.invert_param.chrono_precision = max_prec
  
  U1 = gauge_utils.gaussGauge(latt_size, 0)
  U2 = gauge_utils.unitGauge(latt_size)
  U3 = gauge_utils.unitGauge(latt_size)
  print(f'{U1.data[0, 0, 0, 0, 0, 0]}')
  # print(f'{U3.data[0, 0, 0, 0, 0, 0]}')
  # #my code 
  # U_non_prec = U1.lexico()
  qcu.initGridSize(grid, param, Nc, 1, double_prec, double_prec)
  qcu.loadQcuGauge(U1.data_ptr, 2)		# 2---double 1--float 0---half
  qcu.gauge_reverse_eo_precondition(U2.data_ptr, U1.data_ptr, double_prec)
  qcu.gauge_eo_precondition(U3.data_ptr, U2.data_ptr, double_prec)
  # print(f'U1.data[0, 0, 0, 0, 0, 0] = \n{U1.data[0, 0, 0, 0, 0, 0]}')
  # print(f'U2.data[0, 0, 0, 0, 0, 0] = \n{U2.data[0, 0, 0, 0, 0, 0]}')

  # qcu.eo_precondition(U3.data_ptr, U2.data_ptr, double_prec)
  
  print(f'difference between U1 and U2: \
          , {cp.linalg.norm(U1.data - U2.data) / cp.linalg.norm(U2.data)}')
  print(f'difference between U1 and U3: \
          , {cp.linalg.norm(U1.data - U3.data) / cp.linalg.norm(U3.data)}')

  # print(f'{U1.data[0, 0, 0, 0, 0, 0]}')
  # print(f'{U3.data[0, 0, 0, 0, 0, 0]}')
  qcu.finalizeQcu()

if __name__ == '__main__' :
  test_precondition()