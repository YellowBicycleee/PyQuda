#!/usr/bin/env python3
# this file is modified from test.dslash.qcu.py
# 
import os
import sys
from time import perf_counter

import cupy as cp
import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))

from pyquda import init, core, quda, mpi, pyqcu as qcu
from pyquda.enum_quda import QudaParity
from pyquda.field import LatticeFermion
from pyquda.utils import gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nd, Ns, Nc = 4, 4, 3
latt_size = [16, 16, 32, 32]
grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
print(f'vol = {Vol}')
xi_0, nu = 1, 1
mass=0
coeff_r, coeff_t = 0,0

mpi.init(grid_size)

def test_mpi(round):

  from pyquda.mpi import comm, rank, size, grid, coord, gpuid
  
  p = LatticeFermion(latt_size, cp.random.randn(Lt, Lz, Ly, Lx, Ns, Nc * 2).view(cp.complex128))
 
  Mp1 = LatticeFermion(latt_size)
  Mp2 = LatticeFermion(latt_size)
  x_vector = LatticeFermion(latt_size)
  quda_x = LatticeFermion(latt_size)

  dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False, anti_periodic_t=False) #anti_periodic_t=False 
  U = gauge_utils.gaussGauge(latt_size, 0)
  dslash.loadGauge(U)
  cp.cuda.runtime.deviceSynchronize()
  
  #my code 
  param = qcu.QcuParam()
  grid = qcu.QcuGrid()
  param.lattice_size = latt_size
  grid.grid_size = grid_size
  qcu.initGridSize(grid, param, U.data_ptr, p.even_ptr, Mp1.even_ptr)
  qcu.loadQcuGauge(U.data_ptr, param)
  # then execute my code
  cp.cuda.runtime.deviceSynchronize()

  t1 = perf_counter()
  qcu.cg_inverter(x_vector.even_ptr, p.even_ptr, U.data_ptr, param, 1e-10, 0.125) # Dslash x_vector = Mp1, get x_vector
  t2 = perf_counter()
  #qcu.fullDslashQcu(Mp1.even_ptr, x_vector.even_ptr, U.data_ptr, param, 0) # Dslash x_vector--->Mp2
  quda.MatQuda(Mp1.data_ptr, x_vector.data_ptr, dslash.invert_param)
  print(f'rank {rank} my x and x difference: , {cp.linalg.norm(Mp1.data - p.data) / cp.linalg.norm(Mp1.data)}, takes {t2 - t1} sec')

  t1 = perf_counter()
  print('================quda=================')
  quda.invertQuda(quda_x.data_ptr, p.data_ptr, dslash.invert_param)
  t2 = perf_counter()
  quda.MatQuda(Mp2.data_ptr, quda_x.data_ptr, dslash.invert_param)
  print(f'rank {rank} quda x and x difference: , {cp.linalg.norm(quda_x.data - x_vector.data) / cp.linalg.norm(quda_x.data)}, takes {t2 - t1} sec, norm_quda_x = {cp.linalg.norm(quda_x.data)}')
  print(f'origin_x = {p.data[0, 0, 0, 0, 0]}')
  print(f'quda_x res = {Mp2.data[0, 0, 0, 0, 0]}')
  print(f'my x res = {Mp1.data[0, 0, 0, 0, 0]}')
  
  
  print('============================')

for test in range(0, 1):
    test_mpi(test)


