cdef extern from "qcu.h":
    ctypedef struct QcuParam:
        int lattice_size[4]
    ctypedef struct QcuGrid_t:
        int grid_size[4]
    void initGridSize(QcuGrid_t* grid, QcuParam* p_param, void* gauge, void* fermion_in, void* fermion_out)
    void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)
    void cg_inverter(void* b_vector, void* x_vector, void *gauge, QcuParam *param);
