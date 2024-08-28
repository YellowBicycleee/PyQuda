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
    cdef qcu.QcuGrid_t grid
    def __init__(self):
        pass
    @property
    def grid_size(self):
        return self.grid.grid_size

    @grid_size.setter
    def grid_size(self, value):
        self.grid.grid_size = value

def dslashQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, int parity):
    qcu.dslashQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, parity)

def initGridSize(QcuGrid grid_param, QcuParam param, Pointer gauge, Pointer fermion_in, Pointer fermion_out):
    qcu.initGridSize(&grid_param.grid, &param.param, gauge.ptr, fermion_in.ptr, fermion_out.ptr)

def cg_inverter():
    qcu.cg_inverter(void* b_vector, void* x_vector, void *gauge, QcuParam *param) {

