cimport qcu
from pyquda.pointer cimport Pointer, Pointers, Pointerss

cdef class QcuParam:
    cdef qcu.QcuParam param
    def __init__(self):
        pass

    @property
    def lattice_size(self):
        return self.param.lattice_size

    @lattice_size.setter
    def lattice_size(self, value):
        self.param.lattice_size = value

cdef class QcuGrid:
    cdef qcu.QcuGrid grid
    def __init__(self):
        pass
    @property
    def grid_size(self):
        return self.grid.grid_size

    @grid_size.setter
    def grid_size(self, value):
        self.grid.grid_size = value

# def dslashQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, int parity):
#     qcu.dslashQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, parity)
# 
# def fullDslashQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, int parity):
#     qcu.fullDslashQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, parity)
# #void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag);
# def initGridSize(QcuGrid grid_param, QcuParam param, Pointer gauge, Pointer fermion_in, Pointer fermion_out):
#     qcu.initGridSize(&grid_param.grid, &param.param, gauge.ptr, fermion_in.ptr, fermion_out.ptr)
# 
# 
# def cg_inverter(Pointer b_vector, Pointer x_vector, Pointer gauge, QcuParam param, double p_max_prec, double p_kappa):
#     qcu.cg_inverter(b_vector.ptr, x_vector.ptr, gauge.ptr, &param.param, p_max_prec, p_kappa)
# 
# def loadQcuGauge(Pointer gauge, QcuParam param):
#     qcu.loadQcuGauge(gauge.ptr, &param.param)
# void loadQcuGauge(void* gauge, QcuParam *param);


def initGridSize(QcuGrid grid_param, QcuParam param, int n_color, int m_rhs, int input_precision, int dslash_precision):
    qcu.initGridSize(&grid_param.grid, &param.param, n_color, m_rhs, input_precision, dslash_precision)
    # void initGridSize(QcuGrid *grid, QcuParam *param, int n_color, int m_rhs, int inputFloatPrecision, int dslashFloatPrecision)

def pushBackFermions(Pointer fermion_out, Pointer fermion_in):
    qcu.pushBackFermions(fermion_out.ptr, fermion_in.ptr)
    # void pushBackFermions(void *fermionOut, void *fermionIn)    

def loadQcuGauge(Pointer gauge, int float_precision):
    qcu.loadQcuGauge(gauge.ptr, float_precision)
    # void loadQcuGauge(void *gauge, int floatPrecision)

def getDslash(int dslash_type, double mass):
    qcu.getDslash(dslash_type, mass)
    # void getDslash(int dslashType, double mass)

def finalizeQcu():
    qcu.finalizeQcu()
    # void finalizeQcu()
def start_dslash(int parity, int daggerFlag):
    qcu.start_dslash(parity, daggerFlag)
def mat_Qcu(int daggerFlag):
    qcu.mat_Qcu(daggerFlag)

def qcuInvert(int max_iteration, double p_max_prec):
    qcu.qcuInvert(max_iteration, p_max_prec)

def gauge_eo_precondition(Pointer prec_gauge, Pointer non_prec_gauge, int precision):
    qcu.gauge_eo_precondition(prec_gauge.ptr, non_prec_gauge.ptr, precision)
def gauge_reverse_eo_precondition(Pointer non_prec_gauge, Pointer prec_gauge, int precision):
    qcu.gauge_reverse_eo_precondition(non_prec_gauge.ptr, prec_gauge.ptr, precision)
def read_gauge_from_file (Pointer gauge, bytes binary_file_path_prefix):
    qcu.read_gauge_from_file (gauge.ptr, binary_file_path_prefix);