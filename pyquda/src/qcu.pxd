cdef extern from "qcu.h":
    ctypedef struct QcuParam:
        int lattice_size[4]
    ctypedef struct QcuGrid:
        int grid_size[4]
    void initGridSize(QcuGrid *grid, QcuParam *param, int n_color, int m_rhs, int inputFloatPrecision, int dslashFloatPrecision)
    void pushBackFermions(void *fermionOut, void *fermionIn)
    void loadQcuGauge(void *gauge, int floatPrecision)
    void getDslash(int dslashType, double mass)
    void finalizeQcu()
    void start_dslash(int parity, int daggerFlag)
    void qcuInvert(int max_iteration, double p_max_prec)
    
    void gauge_eo_precondition (void* prec_gauge, void* non_prec_gauge, int precision)
    void gauge_reverse_eo_precondition(void* non_prec_gauge, void* prec_gauge, int precision)
    void read_gauge_from_file (void* gauge, const char* file_path_prefix)