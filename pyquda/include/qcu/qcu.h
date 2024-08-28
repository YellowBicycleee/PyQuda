#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QcuParam_s {
    int lattice_size[4];
} QcuParam;

typedef struct QcuGrid_t {
    int grid_size[4];
} QcuGrid;
void initGridSize(QcuGrid *grid, QcuParam *param, int n_color, int m_rhs, int inputFloatPrecision,
                  int dslashFloatPrecision);
void pushBackFermions(void *fermionOut, void *fermionIn);
void loadQcuGauge(void *gauge, int floatPrecision);  // double precision
void getDslash(int dslashType, double mass);  // dslash precision
void start_dslash(int parity, int daggerFlag);
void qcuInvert(int max_iteration, double p_max_prec);
void finalizeQcu();
#ifdef __cplusplus
}
#endif
