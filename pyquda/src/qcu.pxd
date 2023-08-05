cdef extern from "qcu.h":
    ctypedef struct QcuParam:
        int lattice_size[4]

    void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param)
    void dslashQcuEO(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int even_odd)