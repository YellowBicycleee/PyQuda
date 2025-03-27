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
# latt_size = [4, 4, 4, 4] # lattice description
latt_size = [16, 16, 16, 16] # lattice description
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
    # 128 * color * color + 96 * color 
    vol = latt_desc[0] * latt_desc[1] * latt_desc[2] * latt_desc[3]
    return (128 * color * color + 96 * color) * vol

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

    # profile quda dslash
    t1 = perf_counter()
    for i in range(my_m_input):
        quda.dslashQuda(quda_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
        quda.dslashQuda(quda_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    quda_dslash_time = t2 - t1

    # qcu code 
    qcu.set_tensor_core_flag(0)
    qcu.getDslash(0, mass, 0)           # 参数1：0----WILSON， 参数3暂时未使用
    qcu.loadQcuGauge(U.data_ptr, Precision.kPrecisionDouble)		# 2---double 1--float 0---half

    cp.cuda.runtime.deviceSynchronize()
    qcu_calculate_time = 0
    qcu_scatter_time = 0
    qcu_gather_time = 0
    
    for i in range(my_m_input):
        qcu.pushBackFermions(qcu_Mp_mrhs[i].even_ptr, p_mrhs[i].odd_ptr)
    t1 = perf_counter()
    qcu.begin_gather()
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_gather_time += t2 - t1
    
    t1 = perf_counter()
    qcu.start_dslash(0, 0)	# param1 : parity  param2: dagger
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_calculate_time += t2 - t1

    t1 = perf_counter()
    qcu.begin_scatter()
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_scatter_time += t2 - t1

    for i in range(my_m_input):
        qcu.pushBackFermions(qcu_Mp_mrhs[i].odd_ptr, p_mrhs[i].even_ptr)
    t1 = perf_counter()
    qcu.begin_gather()
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_gather_time += t2 - t1

    t1 = perf_counter()
    qcu.start_dslash(1, 0)
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_calculate_time += t2 - t1

    t1 = perf_counter()
    qcu.begin_scatter()
    cp.cuda.runtime.deviceSynchronize()
    t2 = perf_counter()
    qcu_scatter_time += t2 - t1


    diff = cp.array([cp.linalg.norm(quda_Mp_mrhs[i].data - qcu_Mp_mrhs[i].data) / cp.linalg.norm(quda_Mp_mrhs[i].data) for i in range(my_m_input)])
    if (not warm_flag):
        print(f"Quda dslash: {quda_dslash_time}sec \n"
            f"Qcu dslash: total {qcu_calculate_time + qcu_scatter_time + qcu_gather_time}sec, calculate {qcu_calculate_time}, scatter {qcu_scatter_time}, gather {qcu_gather_time}")

    average_difference = cp.sum(diff) / my_m_input
    print(f'rank {rank}, average difference: {average_difference}')

    return diff


def test_dslash(
        color, 
        m_rhs, 
        input_prec, 
        dslash_prec, 
        warmup_flag = False) :
    
    qcu.initGridSize(grid, param, color, m_rhs, input_prec, dslash_prec)
    
    total_quda_time = 0
    total_qcu_time = 0

    if (not warmup_flag):
        print(f'=========== mrhs = {m_rhs} condition begin ===========')
    # iteration = 1
    # for _ in range(iteration) :
    diff = validate_qcu(m_rhs)
    
    if not warmup_flag :
        print(f'=========== mrhs = {m_rhs} condition end ===========')

    qcu.finalizeQcu()
    cp.cuda.runtime.deviceSynchronize()
    return diff

if __name__ == '__main__' :
    # _ = input()
    num_rhs = 32
    # my_n_color = Nc

    # operations_per_point = (2 * Nd * Nc * Ns) + (2 * Nd * Ns / 2 * (8 * Nc-2)*Nc) + ((2 * Nd - 1) * 2 * Nc * Ns)
    # operations_per_dslash = operations_per_point * Lx * Ly * Lz * Lt

    my_input_prec  = Precision.kPrecisionDouble
    my_dslash_prec = Precision.kPrecisionHalf

    quda_time = []
    qcu_time = []
    qcu_calculate_time = []

    # warm up
    # test_dslash(Nc, 1, input_prec=my_input_prec, dslash_prec=my_dslash_prec, \
    #     quda_average_time = quda_average_time, qcu_average_time = qcu_average_time, warmup_flag=True)
    # warm up end

    half_diff = test_dslash(
        Nc, 
        num_rhs, 
        input_prec=my_input_prec, 
        dslash_prec=Precision.kPrecisionHalf, 
        warmup_flag = False
    )
    print(f'half_diff: {half_diff}')

    float_diff = test_dslash(
        Nc, 
        num_rhs, 
        input_prec=my_input_prec, 
        dslash_prec=Precision.kPrecisionSingle, 
        warmup_flag = False
    )
    print(f'float_diff: {float_diff}')

    double_diff = test_dslash(
        Nc, 
        num_rhs, 
        input_prec=my_input_prec, 
        dslash_prec=Precision.kPrecisionDouble, 
        warmup_flag = False
    )

    def plot_precision_comparison(half_diff, float_diff, double_diff):
        # Convert CuPy arrays to NumPy if necessary
        half_diff = half_diff.get() if hasattr(half_diff, 'get') else half_diff
        float_diff = float_diff.get() if hasattr(float_diff, 'get') else float_diff
        double_diff = double_diff.get() if hasattr(double_diff, 'get') else double_diff
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Set style
        plt.style.use('seaborn')
        
        # Create scatter plot
        x = np.arange(len(half_diff))
        plt.scatter(x, half_diff, label='Half', alpha=0.7, s=50)
        plt.scatter(x, float_diff, label='Single', alpha=0.7, s=50)
        plt.scatter(x, double_diff, label='Double', alpha=0.7, s=50)
        
        # Set log scale for y-axis
        plt.yscale('log')
        
        # Add labels and title
        plt.title('Relative Error Comparison Across Different Precisions', pad=20, fontsize=12)
        plt.xlabel('Sample Index', fontsize=10)
        plt.ylabel('Relative Error', fontsize=10)
        
        # Add grid with custom style
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        plt.legend(fontsize=10)
        
        # Add statistics
        stats_text = (
            f'Statistical Analysis:\n'
            f'Half:   mean={np.mean(half_diff):.2e}, std={np.std(half_diff):.2e}\n'
            f'Single: mean={np.mean(float_diff):.2e}, std={np.std(float_diff):.2e}\n'
            f'Double: mean={np.mean(double_diff):.2e}, std={np.std(double_diff):.2e}'
        )
        plt.figtext(0.15, 0.02, stats_text, fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        # Save figure
        plt.savefig('precision_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 使用示例
    plot_precision_comparison(half_diff, float_diff, double_diff)
