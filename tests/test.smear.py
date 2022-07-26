import os
import sys
from time import perf_counter

import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import quda, core
from pyquda.utils import gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

gauge = gauge_utils.readIldg(os.path.join(test_dir, "weak_field.lime"))

quda.initQuda(0)

core.smear(latt_size, gauge, 1, 0.241)
gauge.setAntiPeroidicT()  # for fermion smearing

quda.endQuda()

gauge_chroma = gauge_utils.readIldg("stout.lime")
print(cp.linalg.norm(gauge.data - gauge_chroma.data))
